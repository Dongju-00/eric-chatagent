"""Build standalone, reviewable notebooks from prepared bundle hashes.

No training or external writes. Drive folder/checkpoint writes happen only when
users execute notebook cells in their own Colab runtime.
"""
import argparse
import hashlib
import json
from pathlib import Path
import textwrap

try:
    from _bootstrap import ensure_src_on_path
except ImportError:
    from model.scripts._bootstrap import ensure_src_on_path
ensure_src_on_path()
from model.paths import PROJECT_ROOT,DATA_DIR


def cell(kind,source):
    result={'cell_type':kind,'metadata':{},'source':textwrap.dedent(source).strip()+'\n'}
    if kind=='code':result.update(execution_count=None,outputs=[])
    return result


SETUP='''
import os, sys, json, math, time, random, hashlib, shutil, zipfile
from pathlib import Path
from datetime import datetime, timezone
from functools import partial
import torch
import sentencepiece as spm
from google.colab import drive

drive.mount('/content/drive', force_remount=False)
DRIVE_ROOT = Path('/content/drive/MyDrive/eric-chatagent')
DATA_DIR = DRIVE_ROOT / 'data'
MODELS_DIR = DRIVE_ROOT / 'models'
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SEED = 20261024
random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', DEVICE, '| model outputs:', MODELS_DIR)
'''

DATA='''
BUNDLE_NAME = __BUNDLE_NAME__
EXPECTED_BUNDLE_SHA256 = __BUNDLE_HASH__
BUNDLE = DATA_DIR / BUNDLE_NAME
if not BUNDLE.is_file() and (DRIVE_ROOT / BUNDLE_NAME).is_file():
    shutil.copyfile(DRIVE_ROOT / BUNDLE_NAME, BUNDLE)
if not BUNDLE.is_file():
    from google.colab import files
    print('Upload exactly:', BUNDLE_NAME)
    uploaded = files.upload()
    if BUNDLE_NAME not in uploaded:
        raise FileNotFoundError('Expected prepared bundle: ' + BUNDLE_NAME)
    shutil.copyfile(Path('/content') / BUNDLE_NAME, BUNDLE)
assert hashlib.sha256(BUNDLE.read_bytes()).hexdigest() == EXPECTED_BUNDLE_SHA256, 'Wrong/stale bundle; use the archive accompanying this notebook'
EXTRACT_DIR = Path('/content') / (TASK + '_finetune_' + EXPECTED_BUNDLE_SHA256[:12])
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(BUNDLE) as archive:
    for name in archive.namelist():
        if not (EXTRACT_DIR / name).resolve().is_relative_to(EXTRACT_DIR.resolve()):
            raise ValueError('Unsafe ZIP entry')
    archive.extractall(EXTRACT_DIR)
manifest = json.loads((EXTRACT_DIR / 'manifest.json').read_text())
assert manifest['task'] == TASK
SP_MODEL = EXTRACT_DIR / 'sp_korean.model'
assert hashlib.sha256(SP_MODEL.read_bytes()).hexdigest() == manifest['tokenizer_sha256']
sp = spm.SentencePieceProcessor(model_file=str(SP_MODEL))
sys.path.insert(0, str(EXTRACT_DIR / 'runtime'))
from finetune_data import answer_mask, collate_samples, causal_loss, evaluate_token_weighted, save_torch_state
from model_definition import KoreanGPT, block_size
assert block_size == manifest['model_block_size'] == 384
train_records = [json.loads(line) for line in (EXTRACT_DIR / 'train.jsonl').read_text().splitlines()]
val_records = [json.loads(line) for line in (EXTRACT_DIR / 'val.jsonl').read_text().splitlines()]
assert train_records and val_records
assert not {r['source_key'] for r in train_records} & {r['source_key'] for r in val_records}, 'Source-level split leakage'

def encode_records(records):
    samples = []
    for record in records:
        ids = sp.encode(record['text'], out_type=int)
        assert len(ids) == record['token_count'] <= block_size, 'Token budget changed'
        mask = answer_mask(ids, sp)
        assert any(mask[1:]), 'No shifted answer supervision'
        samples.append({'input_ids': ids, 'answer_mask': mask})
    return samples

train_samples = encode_records(train_records)
val_samples = encode_records(val_records)
print('source overlap: 0 | train:', len(train_samples), '| val:', len(val_samples))
for name, samples in [('train', train_samples), ('val', val_samples)]:
    lengths = sorted(len(s['input_ids']) for s in samples)
    print(name, 'tokens:', sum(lengths), 'median:', lengths[len(lengths)//2], 'max:', lengths[-1],
          'answer targets:', sum(sum(s['answer_mask'][1:]) for s in samples))
print(json.dumps(manifest, ensure_ascii=False, indent=2))
'''

MODEL='''
from huggingface_hub import hf_hub_download
PRETRAINED_NAME = __PRETRAINED__
OUTPUT_NAME = __OUTPUT__
# Reuse an explicitly placed original model; otherwise download only that weight file.
CKPT_IN = MODELS_DIR / PRETRAINED_NAME
if not CKPT_IN.is_file():
    CKPT_IN = Path(hf_hub_download('Dongju-00/eric-chatagent',
        filename='src/model/checkpoints/' + PRETRAINED_NAME,
        local_dir=str(MODELS_DIR / 'pretrained')))
model = KoreanGPT(sp.get_piece_size()).to(DEVICE)
state = torch.load(CKPT_IN, map_location='cpu', weights_only=True)
model.load_state_dict(state)
del state
assert model.pos_emb.num_embeddings == 384
print('Loaded:', CKPT_IN, '| parameters:', sum(p.numel() for p in model.parameters()))
'''

LOADERS='''
from torch.utils.data import DataLoader
CONFIG = {
    'task': TASK, 'train_window_target': __WINDOW__, 'model_block_size': 384,
    'batch_size': __BATCH__, 'max_epochs': 2, 'lr': __LR__,
    'patience': 1, 'min_delta': 1e-4, 'warmup_ratio': .05,
    'weight_decay': .01, 'grad_clip': 1., 'seed': SEED,
    'bundle_sha256': EXPECTED_BUNDLE_SHA256,
}
torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
loader_generator = torch.Generator().manual_seed(SEED)
collate = partial(collate_samples, pad_id=max(0, sp.pad_id()), max_length=384)
train_dl = DataLoader(train_samples, batch_size=CONFIG['batch_size'], shuffle=True,
    generator=loader_generator, drop_last=False, collate_fn=collate, num_workers=0)
val_dl = DataLoader(val_samples, batch_size=CONFIG['batch_size'], shuffle=False,
    collate_fn=collate, num_workers=0)
assert len(train_dl) > 0 and len(val_dl) > 0
x, y = next(iter(val_dl))
assert x.shape == y.shape and x.shape[1] <= 384
assert y[:,1:].ne(-100).any()
with torch.no_grad():
    model.eval()
    smoke_loss = causal_loss(model, x[:1].to(DEVICE), y[:1].to(DEVICE))
assert torch.isfinite(smoke_loss), 'Invalid masked loss'
print('Finite loss probe:', smoke_loss.item(), '| steps per epoch:', len(train_dl))
print(json.dumps(CONFIG, ensure_ascii=False, indent=2))
'''


METRICS_SETUP="""
EVALUATION_CASES = build_evaluation_cases(val_records, TASK, limit=16, seed=SEED)
EVALUATION_SHA256 = protocol_sha256(EVALUATION_CASES)
CONFIG['evaluation_protocol_sha256'] = EVALUATION_SHA256
CONFIG['evaluation_code_sha256'] = __METRICS_SHA__
CONFIG['evaluation_max_new_tokens'] = 96 if TASK == 'news' else 80
DATASET_AUDIT = {'train': audit_multiturn(train_records), 'val': audit_multiturn(val_records)}
assert DATASET_AUDIT['train']['multiturn_samples'] > 0, 'No multi-turn training examples'
assert DATASET_AUDIT['val']['multiturn_samples'] > 0, 'No multi-turn validation examples'
print('Multi-turn training audit:', json.dumps(DATASET_AUDIT, ensure_ascii=False, indent=2))
print('Frozen evaluation cases:', len(EVALUATION_CASES), '| protocol:', EVALUATION_SHA256)
"""

TRAIN='''
assert DEVICE == 'cuda', 'Use a Colab GPU runtime; CPU training is intentionally disabled'
# Set this only to this notebook's own trusted resume file, never an untrusted pickle.
RESUME_PATH = None
if RESUME_PATH:
    RUN_DIR = Path(RESUME_PATH).parent
else:
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    RUN_DIR = MODELS_DIR / (TASK + '_' + run_id)
    RUN_DIR.mkdir(parents=True, exist_ok=False)
CKPT_OUT = RUN_DIR / OUTPUT_NAME
LAST_STATE = RUN_DIR / 'last.resume.pt'
HISTORY_PATH = RUN_DIR / 'training_history.json'
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
total_steps = len(train_dl) * CONFIG['max_epochs']
warmup = max(1, int(total_steps * CONFIG['warmup_ratio']))
def lr_factor(step):
    if step < warmup: return (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return .5 * (1 + math.cos(math.pi * min(1., progress)))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
start_epoch = 0; stale = 0; history = []
baseline_loss = None
best = None
if RESUME_PATH:
    resume = torch.load(RESUME_PATH, map_location='cpu', weights_only=False)
    assert resume['config'] == CONFIG, 'Resume config/data mismatch'
    assert CKPT_OUT.is_file(), 'Best model is missing from the resumed run'
    model.load_state_dict(resume['model'])
    optimizer.load_state_dict(resume['optimizer']); scheduler.load_state_dict(resume['scheduler'])
    start_epoch = resume['next_epoch']; best = resume['best']; stale = resume['stale']
    baseline_loss = resume['baseline_loss']; history = resume['history']
    random.setstate(resume['python_rng']); torch.set_rng_state(resume['torch_rng'])
    torch.cuda.set_rng_state_all(resume['cuda_rng'])
    loader_generator.set_state(resume['loader_rng'])
else:
    # Always measure the actual original checkpoint, including cell re-runs.
    model.load_state_dict(torch.load(CKPT_IN, map_location='cpu', weights_only=True))
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    loader_generator.manual_seed(SEED)
    baseline_loss = evaluate_token_weighted(model, val_dl, DEVICE)
    best = baseline_loss
    # Preserve the baseline as best if no fine-tuned epoch improves validation.
    save_torch_state({k: v.detach().cpu() for k,v in model.state_dict().items()}, CKPT_OUT)
    (RUN_DIR / 'config.json').write_text(json.dumps(CONFIG, indent=2), encoding='utf-8')

BEFORE_PATH = RUN_DIR / 'before_metrics.json'
AFTER_PATH = RUN_DIR / 'after_metrics.json'
if RESUME_PATH:
    assert BEFORE_PATH.is_file(), 'This run has no original before measurement; use a fresh run for a valid comparison'
    before_metrics = json.loads(BEFORE_PATH.read_text())
    assert before_metrics['protocol_sha256'] == EVALUATION_SHA256
else:
    save_json(RUN_DIR / 'evaluation_cases.json', EVALUATION_CASES)
    save_json(RUN_DIR / 'dataset_audit.json', DATASET_AUDIT)
    before_metrics = measure_model(model, sp, EVALUATION_CASES, DEVICE, answer_mask, TASK,
        baseline_loss, max_new_tokens=CONFIG['evaluation_max_new_tokens'], seed=SEED)
    before_metrics['checkpoint_sha256'] = file_sha256(CKPT_IN)
    before_metrics['evaluation_code_sha256'] = CONFIG['evaluation_code_sha256']
    save_json(BEFORE_PATH, before_metrics)
print('Baseline val loss:', baseline_loss, '| output:', CKPT_OUT)
print('Persisted original-model metrics:', BEFORE_PATH)
for epoch in range(start_epoch, CONFIG['max_epochs']):
    if stale >= CONFIG['patience']: break
    model.train(); started = time.time(); running = 0.; targets = 0
    optimizer.zero_grad(set_to_none=True)
    for step, (x,y) in enumerate(train_dl):
        x=x.to(DEVICE); y=y.to(DEVICE)
        loss=causal_loss(model,x,y)
        if not torch.isfinite(loss): raise ValueError('Non-finite loss; stop without replacing best')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),CONFIG['grad_clip'])
        optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        count=int(y[:,1:].ne(-100).sum()); running+=loss.item()*count; targets+=count
        if step % 100 == 0: print('epoch',epoch+1,'step',step,'loss',round(loss.item(),4),flush=True)
    val_loss=evaluate_token_weighted(model,val_dl,DEVICE)
    improved=val_loss < best-CONFIG['min_delta']
    if improved:
        best=val_loss; stale=0
        save_torch_state({k:v.detach().cpu() for k,v in model.state_dict().items()},CKPT_OUT)
    else: stale+=1
    history.append({'epoch':epoch+1,'train_loss':running/targets,'val_loss':val_loss,
                    'improved':improved,'seconds':time.time()-started})
    HISTORY_PATH.write_text(json.dumps(history,indent=2),encoding='utf-8')
    save_torch_state({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
        'scheduler':scheduler.state_dict(),'next_epoch':epoch+1,'best':best,'stale':stale,
        'baseline_loss':baseline_loss,'history':history,'config':CONFIG,
        'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),
        'cuda_rng':torch.cuda.get_rng_state_all(),'loader_rng':loader_generator.get_state()},LAST_STATE)
    print(history[-1], '| best:',CKPT_OUT,flush=True)

model.load_state_dict(torch.load(CKPT_OUT,map_location='cpu',weights_only=True))
print('Reloaded best state:',CKPT_OUT,'| baseline:',baseline_loss,'best:',best)
best_sha256 = file_sha256(CKPT_OUT)
if AFTER_PATH.is_file():
    after_metrics = json.loads(AFTER_PATH.read_text())
    assert after_metrics['checkpoint_sha256'] == best_sha256, 'Stored evaluation does not match best model'
else:
    after_loss = evaluate_token_weighted(model, val_dl, DEVICE)
    after_metrics = measure_model(model, sp, EVALUATION_CASES, DEVICE, answer_mask, TASK,
        after_loss, max_new_tokens=CONFIG['evaluation_max_new_tokens'], seed=SEED)
    after_metrics['checkpoint_sha256'] = best_sha256
    after_metrics['evaluation_code_sha256'] = CONFIG['evaluation_code_sha256']
    save_json(AFTER_PATH, after_metrics)
comparison = save_comparison(RUN_DIR, before_metrics, after_metrics)
save_json(RUN_DIR / 'training_summary.json', {
    'baseline_validation_nll': baseline_loss, 'best_validation_nll': best,
    'delta_best_minus_baseline': best-baseline_loss, 'epochs_completed': len(history),
    'optimizer_steps_completed': len(history)*len(train_dl),
    'training_seconds': sum(h['seconds'] for h in history),
    'config': CONFIG, 'best_checkpoint_sha256': best_sha256,
    'comparison_file': str(RUN_DIR / 'comparison.json')})
print('Saved before/after metrics and comparison:', RUN_DIR / 'comparison.csv')
if best >= baseline_loss-CONFIG['min_delta']:
    print('Validation did not improve: best output retains the initial weights; do not claim an improvement.')
'''

EVALUATE="""
print('Before / best / after-minus-before:')
for key, value in comparison['delta_after_minus_before'].items():
    print(key, '|', comparison['before'].get(key), '|', comparison['after'].get(key), '| delta:', value)
print('Semantic accuracy is not automatically certified. Review manual_review.csv.')
print('Full artifacts:', RUN_DIR)
for old,new in zip(before_metrics['rows'], after_metrics['rows']):
    if old['category']=='controlled_reference':
        print(old['case_id'], '| expected anchor:', old['expected_anchor'],
              '| before:', old['generated_answer'], '| best:', new['generated_answer'])
"""



def build(task,bundle,output):
    bundle_hash=hashlib.sha256(bundle.read_bytes()).hexdigest()
    is_news=task=='news'
    pretrained='KoreanGPT_news_v2.pt' if is_news else 'KoreanGPT_smalltalk.pt'
    final='KoreanGPT_news_v3.pt' if is_news else 'KoreanGPT_smalltalk_mt.pt'
    title='뉴스' if is_news else '스몰톡'
    data=DATA.replace('__BUNDLE_NAME__',repr(bundle.name)).replace('__BUNDLE_HASH__',repr(bundle_hash))
    model=MODEL.replace('__PRETRAINED__',repr(pretrained)).replace('__OUTPUT__',repr(final))
    loaders=LOADERS.replace('__WINDOW__','384' if is_news else '256').replace('__BATCH__','16' if is_news else '32').replace('__LR__','1e-5' if is_news else '2e-5')
    evaluation=EVALUATE
    metrics_source=(PROJECT_ROOT/'src/model/train/finetune_metrics.py').read_text()
    metrics_hash=hashlib.sha256(metrics_source.encode()).hexdigest()
    metrics_setup=METRICS_SETUP.replace('__METRICS_SHA__',repr(metrics_hash))
    cells=[cell('markdown',f'''# {title} 멀티턴 fine-tuning — 검증된 데이터 묶음 사용

모델의 `block_size=384`와 vocab16000은 유지합니다. {'실제 신규 기사 600건 중 본문 품질을 확인한 587건의 4턴 QA, 기사별 528/59 분리입니다. 나머지13건은 불완전한 문장/본문 추출 문제로 제외했습니다. 답변은 원문 인용, 질문은 후속 질문 템플릿입니다. 기존 QA 재배열본으로 자동 대체하지 않습니다.' if is_news else '학습 샘플은 최대256토큰이며 QA 중간을 자르지 않습니다. 원본 세션과 합성 대화 템플릿을 기준으로 train/val을 나눠 겹치는 윈도우의 누출을 막았습니다.'}

짧은 토큰 윈도우 자체가 이해력을 높인다고 보장하지 않습니다. 대화 이력을 유지하고 답변만 학습하며, 검증 손실과 단일턴/멀티턴 비교로 채택합니다.

**최대2에폭**, 검증 손실 미개선1회 시 조기 종료. 모델 파일은 **MyDrive/eric-chatagent/models/** 아래 새 실행 폴더에 저장합니다. 준비 코드의 로컬 점검과 Colab GPU 학습 실행은 별개입니다. 아래 셀을 순서대로 실행하세요.'''),
        cell('code','!pip -q install sentencepiece huggingface_hub'),
        cell('code',f'TASK = {task!r}\n'+SETUP),
        cell('markdown',f'## 데이터 준비\n`{bundle.name}`를 Drive의 `eric-chatagent/data/`에 넣으세요. 없으면 이 ZIP 한 개만 업로드합니다. 이전 CSV/캐시는 사용하지 않습니다.'),
        cell('code',data),cell('markdown','## 기존 모델 적재\n초기 모델은 별도로 보존하고, 검증한 묶음의 토크나이저를 사용합니다.'),
        cell('code',model),cell('markdown','## 학습 설정과 손실 검사\n배치별 동적 패딩, 질문/컨텍스트/패딩 마스킹, 다음 토큰 shift를 적용합니다. 검증 손실은 답변 토큰 수로 가중합니다.'),
        cell('code',loaders),cell('markdown','## 멀티턴 학습 검사와 고정 평가셋\n동일한 평가셋으로 기존 모델과 best 모델을 비교합니다. 이전 대화를 제거한 likelihood, 대화 이력만 바꾼 지시대명사 대조쌍, 단일턴 퇴행 및 뉴스 거절 표현을 기록합니다. 자동 점수는 의미 정확도의 증명이 아닙니다.'),
        cell('code',metrics_source),cell('code',metrics_setup),cell('markdown','## GPU 학습 및 Drive 저장\nA100 또는 사용 가능한 Colab GPU로 실행하세요. 매 에폭 best 가중치와 재개 상태를 Drive에 저장합니다. 중단 후 `RESUME_PATH`에 해당 실행의 `last.resume.pt`를 지정하면 에폭 경계부터 재개합니다.'),
        cell('code',TRAIN),cell('markdown','## best 체크포인트 비교\n마지막 에폭이 아닌 검증상 best 가중치를 사용합니다. 근거 없는 수치를 만들거나 단일턴이 나빠지면 채택하지 마세요. 전후 수치와 원문 답변은 Drive의 comparison.json/CSV와 manual_review.csv에 저장합니다. 자동 점수는 일반화 성능을 보장하지 않습니다.'),
        cell('code',evaluation)]
    for index, item in enumerate(cells): item['id'] = f'{task}-{index:02d}'
    notebook={'cells':cells,'metadata':{'colab':{'name':f'eric_{task}_multiturn_colab.ipynb'},
                'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},
              'nbformat':4,'nbformat_minor':5}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return {'task':task,'notebook':str(output),'bundle':str(bundle),'bundle_sha256':bundle_hash}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-dir',type=Path,default=DATA_DIR/'colab_finetune_curated_2026-10-24')
    parser.add_argument('--output-dir',type=Path,default=PROJECT_ROOT/'notebooks/colab')
    args=parser.parse_args()
    for task in ['news','smalltalk']:
        print(json.dumps(build(task,args.bundle_dir/f'{task}_finetune_colab.zip',args.output_dir/f'eric_{task}_multiturn_colab.ipynb'),ensure_ascii=False))

if __name__=='__main__':main()
