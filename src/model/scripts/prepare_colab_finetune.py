"""Build reviewed Colab data from new article bodies and source-grouped smalltalk.

Does not train models, modify source data, touch checkpoints or use the network.
All generated files are exclusive new artifacts. Evidence is extractive and
questions are authored follow-up templates, not independent human conversations.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace
import zipfile

import sentencepiece as spm
try:
    from _bootstrap import ensure_src_on_path
except ImportError:
    from model.scripts._bootstrap import ensure_src_on_path
ensure_src_on_path()
from model.paths import DATA_DIR,MODEL_ROOT,SP_PREFIX
from model.scripts.prepare_multiturn_news import (
    RECOMMENDATION,SUPPORTED_ANSWER_TEMPLATES,REFUSAL,clean_text,normalize_title,
    existing_titles,validate_conversation,write_dataset,render_conversation,split_articles)
from model.train.finetune_data import parse_pairs,render_pairs,window_pairs,split_groups,answer_mask


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def excerpt_candidates(source):
    body=source['body']
    sentences=list(re.finditer(r'.+?(?:[.!?](?=\s|$)|$)',body))
    result=[]
    finance=re.compile(r'실적|매출|영업이익|투자|주가|성장|사업|메모리|수출|금리|생산|공급|시장|수요')
    for first,second in zip(sentences,sentences[1:]):
        a,b=first.group().strip(),second.group().strip()
        context=body[first.start():second.end()].strip()
        if 8<=len(a)<=100 and 8<=len(b)<=100 and 50<=len(context)<=170 and not RECOMMENDATION.search(context):
            score=len(finance.findall(context)) - first.start()/max(1,len(body))
            result.append((score,context,a,b,'complete_sentences'))
    # Real excerpts, never invented replacement news. For long sentences, use
    # word-aligned contiguous clauses and disclose that selection in provenance.
    if not result:
        words=list(re.finditer(r'\S+',body))
        for i in range(len(words)):
            for j in range(i+1,min(i+25,len(words))):
                a=body[words[i].start():words[j-1].end()]
                if not 25<=len(a)<=80:continue
                for k in range(j+1,min(j+25,len(words))):
                    b=body[words[j].start():words[k-1].end()]
                    context=body[words[i].start():words[k-1].end()]
                    if 25<=len(b)<=80 and 60<=len(context)<=165 and not RECOMMENDATION.search(context):
                        result.append((len(finance.findall(context))-i/max(1,len(words)),context,a,b,'word_aligned_excerpts'))
                        break
                if result:break
            if result:break
    return sorted(result,key=lambda r:r[0],reverse=True)


FIRST_QUESTIONS=['이 뉴스에서 먼저 확인되는 내용은 뭐야?','이 기사의 핵심 내용을 알려줘.','기사에서는 어떤 소식을 전하고 있어?',
                 '이 소식에서 확인할 수 있는 내용을 짚어줘.','이 뉴스는 어떤 내용이야?','기사에 나온 첫 내용을 간단히 알려줘.']
SECOND_QUESTIONS=['방금 말한 내용에 이어 기사에서는 무엇을 설명해?','그 소식에서 함께 확인할 수 있는 내용도 알려줘.',
                  '앞서 말한 내용 다음에는 어떤 설명이 나와?','그 내용과 함께 기사에 실린 문장을 더 보여줘.',
                  '방금 설명한 소식에 덧붙여진 내용은 뭐야?','그 소식에서 추가로 확인되는 부분은?']
TITLE_QUESTIONS=['그럼 방금 말한 소식의 제목에서 어떤 표현이 나와?','그 내용이 실린 기사의 제목을 짚어줘.',
                 '앞서 설명한 뉴스의 제목에 나온 표현도 확인해줘.']
PRIVATE_QUESTIONS=['그럼 내 증권 계좌에 있는 보유 수량도 나와?','그 소식만으로 내 계좌의 평균 매입 단가도 알 수 있어?',
                   '앞서 말한 기사에 내 계좌의 보유 비중도 나와?']
FUTURE_QUESTIONS=['그럼 그 소식만으로 다음 거래일의 실제 종가도 알 수 있어?',
                  '방금 말한 뉴스에 다음 거래일의 실제 장중 최고 가격도 나와?',
                  '그 내용만으로 다음 거래일의 확정된 최저 가격도 확인돼?']


def news_conversation(source,index,total,sp):
    extras=round(total*4*.28)-total
    refusals=1+int((index+1)*extras//total > index*extras//total)
    title_evidence=source['title'] if len(source['title'])<=90 else source['title'][:90].rsplit(' ',1)[0]
    for _,context,a,b,method in excerpt_candidates(source):
        def supported(question,evidence,number):
            return {'question':question,'answer':SUPPORTED_ANSWER_TEMPLATES[number%3].format(evidence=evidence),
                    'evidence':evidence,'unanswerable':False}
        def unknown(question):return {'question':question,'answer':REFUSAL+'.','evidence':'','unanswerable':True}
        third = supported(TITLE_QUESTIONS[index%3],title_evidence,index+2)
        if RECOMMENDATION.search(title_evidence):
            third = supported('그럼 앞에서 알려준 핵심 문장을 다시 확인해줘.',a,index+2)
        turns=[supported(FIRST_QUESTIONS[index%6],a,index),supported(SECOND_QUESTIONS[index%6],b,index+1),
               unknown(PRIVATE_QUESTIONS[index%3]) if refusals==2 else third,
               unknown(FUTURE_QUESTIONS[index%3])]
        candidate={'source_id':source['source_id'],'context':'제목: '+source['title']+'\n내용: '+context,
                   'turns':turns,'generation_method':'extractive_followup_v1','excerpt_method':method}
        try:
            validate_conversation(candidate,source,sp,refusals,expected_turns=4)
            return candidate
        except ValueError:
            continue
    raise ValueError('No grounded in-budget excerpt found: '+source['source_id'])


def write_jsonl(path,records):
    with path.open('x',encoding='utf-8') as f:
        for record in records:f.write(json.dumps(record,ensure_ascii=False)+'\n')


def make_record(identity,text,source_key,sp,**extra):
    ids=sp.encode(text,out_type=int)
    mask=answer_mask(ids,sp)
    if len(ids)>384 or not any(mask[1:]):raise ValueError('Invalid training sample '+identity)
    return {'conversation_id':identity,'source_key':source_key,'text':text,'token_count':len(ids),
            'answer_tokens':sum(mask[1:]),**extra}


def pack_bundle(path,task,train,val,sp_path,manifest,extra_files=()):
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,records in [('train.jsonl',train),('val.jsonl',val)]:
            z.writestr(name,''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
        z.write(sp_path,'sp_korean.model')
        z.write(MODEL_ROOT/'train/model.py','runtime/model_definition.py')
        z.write(MODEL_ROOT/'train/finetune_data.py','runtime/finetune_data.py')
        z.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        z.writestr('README.txt',f'{task}: upload this archive to MyDrive/eric-chatagent/data/. Open the reviewed {task} notebook. Models are saved under eric-chatagent/models/. Original model block_size384, tokenizer unchanged. No model training has been executed by data preparation.\n')
        for file in extra_files:z.write(file,file.name)
    with zipfile.ZipFile(path) as z:
        if z.testzip():raise ValueError('Corrupt bundle')


def build_news(output,sp,sp_path):
    source_path=DATA_DIR/'news/multiturn_collection/articles.jsonl'
    source_hash=digest(source_path)
    sources=[json.loads(line) for line in source_path.read_text().splitlines() if line.strip()]
    if len(sources)!=600:raise ValueError('Expected exactly 600 collected articles')
    excluded=existing_titles(DATA_DIR/'news/qa_news_train_merged.csv')
    titles=set();ids=set();bodies=set();urls=set()
    for source in sources:
        title=normalize_title(source['title']);body_hash=hashlib.sha256(clean_text(source['body']).encode()).hexdigest()
        if body_hash!=source['body_sha256'] or source['source_id'] in ids or title in titles|excluded or body_hash in bodies or source['url'] in urls:
            raise ValueError('Invalid article provenance or overlap')
        titles.add(title);ids.add(source['source_id']);bodies.add(body_hash);urls.add(source['url'])
    rejected=[];curated=[]
    for source in sources:
        complete=[c for c in excerpt_candidates(source) if c[-1]=='complete_sentences']
        if not complete:
            rejected.append({'source_id':source['source_id'],'title':source['title'],
                             'reason':'No two complete short evidence sentences; fallback may capture boilerplate or captions'})
        else:curated.append(source)
    sources=curated
    if len(sources)<500:raise ValueError('Too few complete article contexts after quality filtering')
    conversations={source['source_id']:news_conversation(source,index,len(sources),sp) for index,source in enumerate(sources)}
    args=SimpleNamespace(target=len(sources),minimum_articles=500,turns_per_conversation=4,existing=DATA_DIR/'news/qa_news_train_merged.csv',
                         work_dir=source_path.parent,output_dir=DATA_DIR/'news/curated_2026-10-24',tokenizer=sp_path)
    write_dataset(args,sources,conversations,sp)
    report=json.loads((args.output_dir/'qa_news_multiturn.report.json').read_text())
    source_train,source_val=split_articles(sources)
    def records(group):return [make_record(s['source_id'],render_conversation(conversations[s['source_id']]),
                                           'article:'+s['source_id'],sp,turns=4) for s in group]
    train,val=records(source_train),records(source_val)
    manifest={**report,'task':'news','train_window_target':384,'model_block_size':384,'source_articles_received':600,'source_articles':len(sources),'quality_exclusions':rejected,
              'generation_method':'extractive_followup_v1','article_cache_sha256':source_hash,
              'excerpt_methods':dict(Counter(c['excerpt_method'] for c in conversations.values())),
              'tokenizer_sha256':digest(sp_path),'token_count':sum(r['token_count'] for r in train+val),
              'answer_tokens':sum(r['answer_tokens'] for r in train+val)}
    write_jsonl(output/'news_conversations.jsonl',conversations.values())
    (args.output_dir/'qa_news_multiturn.report.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    pack_bundle(output/'news_finetune_colab.zip','news',train,val,sp_path,manifest,
                [args.output_dir/'qa_news_multiturn_train.csv',args.output_dir/'qa_news_multiturn_val.csv'])
    assert source_hash==digest(source_path)
    return manifest


def build_smalltalk(output,sp,sp_path):
    text_path=DATA_DIR/'processed/smalltalk_multiturn.txt'
    metadata_path=DATA_DIR/'processed/smalltalk_multiturn.manifest.jsonl'
    input_hash=digest(text_path)
    blocks=text_path.read_text().rstrip('\n').split('\n\n')
    metadata=[json.loads(line) for line in metadata_path.read_text().splitlines()]
    if len(blocks)!=len(metadata):raise ValueError('Smalltalk manifest mismatch')
    sessions={};last_sizes={};last_indices={}
    for text,meta in zip(blocks,metadata):
        if meta['source_type']=='aihub':key='aihub:'+meta['source_file']
        else:key='synthetic:'+str(meta['template_index'])+':'+meta['session_id']
        pairs=parse_pairs(text)
        if key in sessions:
            if meta['window_index']!=last_indices[key]+1:raise ValueError('Non-contiguous source windows')
            if last_sizes[key]>=2:
                if sessions[key][-1]!=pairs[0]:raise ValueError('Window overlap mismatch')
                pairs=pairs[1:]
            sessions[key].extend(pairs)
        else:sessions[key]=pairs
        last_sizes[key]=meta['turns'];last_indices[key]=meta.get('window_index',0)
    def split_key(key):return ':'.join(key.split(':')[:2]) if key.startswith('synthetic:') else key
    train_keys,val_keys=split_groups([split_key(k) for k in sessions],.05)
    train=[];val=[];exceptions=0;max_single=0
    for key,pairs in sessions.items():
        largest=max(len(sp.encode(render_pairs([p])+'\n')) for p in pairs)
        limit=256 if largest<=256 else 384
        if limit>256:exceptions+=1
        max_single=max(max_single,largest)
        for i,window in enumerate(window_pairs(pairs,sp,limit)):
            text=render_pairs(window)+'\n'
            record=make_record(key+':'+str(i),text,split_key(key),sp,turns=len(window),window_target=limit)
            (train if split_key(key) in train_keys else val).append(record)
    if {r['source_key'] for r in train}&{r['source_key'] for r in val}:raise ValueError('Source split leakage')
    manifest={'task':'smalltalk','train_window_target':256,'model_block_size':384,'source_sessions':len(sessions),
              'train_samples':len(train),'val_samples':len(val),'source_group_overlap':0,
              'extended_context_sessions':exceptions,'largest_complete_qa_tokens':max_single,
              'token_count':sum(r['token_count'] for r in train+val),'answer_tokens':sum(r['answer_tokens'] for r in train+val),
              'max_sample_tokens':max(r['token_count'] for r in train+val),'input_text_sha256':input_hash,
              'tokenizer_sha256':digest(sp_path),'split_unit':'original AIHub source or synthetic authored template',
              'note':'Single QA pairs >256 are preserved under the existing 384-token model limit; never silently truncated.'}
    pack_bundle(output/'smalltalk_finetune_colab.zip','smalltalk',train,val,sp_path,manifest)
    assert input_hash==digest(text_path)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=DATA_DIR/'colab_finetune_curated_2026-10-24')
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if any(args.output_dir.iterdir()):raise FileExistsError('Use a new empty output directory')
    sp_path=Path(str(SP_PREFIX)+'.model');tokenizer_hash=digest(sp_path)
    sp=spm.SentencePieceProcessor(model_file=str(sp_path))
    reports={'news':build_news(args.output_dir,sp,sp_path),'smalltalk':build_smalltalk(args.output_dir,sp,sp_path)}
    if digest(sp_path)!=tokenizer_hash:raise ValueError('Tokenizer changed')
    with (args.output_dir/'preparation_report.json').open('x',encoding='utf-8') as f:json.dump(reports,f,ensure_ascii=False,indent=2)
    print(json.dumps(reports,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
