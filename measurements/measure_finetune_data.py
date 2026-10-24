"""Record data-side multi-turn readiness, without measuring model performance.

python measurements/measure_finetune_data.py --output measurements/results/<label>.json
Before/best model scores are measured in Colab and must never be inferred here.
"""
import argparse
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'src'))
from model.paths import DATA_DIR
from model.train.finetune_metrics import audit_multiturn,build_evaluation_cases,protocol_sha256


def collect(bundle_dir):
    result={'model_performance_measured':False,
            'note':'Actual data structure measurements only. Before/best model metrics are collected by Colab execution, not inferred from these counts.'}
    for task in ['news','smalltalk']:
        with zipfile.ZipFile(bundle_dir/f'{task}_finetune_colab.zip') as archive:
            if archive.testzip():raise ValueError('Invalid ZIP')
            train=[json.loads(s) for s in archive.read('train.jsonl').decode().splitlines()]
            val=[json.loads(s) for s in archive.read('val.jsonl').decode().splitlines()]
        if {r['source_key'] for r in train}&{r['source_key'] for r in val}:raise ValueError('Source overlap')
        cases=build_evaluation_cases(val,task)
        result[task]={'train':audit_multiturn(train),'val':audit_multiturn(val),
                      'actual_model_window':384,'sample_max_tokens':max(r['token_count'] for r in train+val),
                      'total_tokens':sum(r['token_count'] for r in train+val),
                      'answer_tokens':sum(r['answer_tokens'] for r in train+val),
                      'evaluation_cases':len(cases),'evaluation_protocol_sha256':protocol_sha256(cases)}
        if task=='smalltalk':
            previous=json.loads((DATA_DIR/'processed/smalltalk_multiturn.report.json').read_text())
            result[task]['previous_prepared_text']={key:previous[key] for key in ['output_fragments','output_turns','total_tokens','max_serialized_tokens']}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-dir',type=Path,default=DATA_DIR/'colab_finetune_curated_2026-10-24')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Use a new output label; previous measurements are retained')
    result=collect(args.bundle_dir)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
