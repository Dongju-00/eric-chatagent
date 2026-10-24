"""Read-only independent full-corpus validation of smalltalk data artifacts."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import re
from pathlib import Path
import sentencepiece as spm
import argparse
try:
 from _bootstrap import ensure_src_on_path
except ImportError:
 from model.scripts._bootstrap import ensure_src_on_path
ensure_src_on_path()
from model.paths import DATA_DIR


def main():
    parser=argparse.ArgumentParser(description='Independently verify generated smalltalk text, gzip, metadata and source hashes.')
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--report', type=Path, help='Optional new report path; never overwritten')
    args=parser.parse_args()
    data=args.data_dir
    if args.report and args.report.exists():
     raise FileExistsError('Verification report already exists; choose a new path.')
    if not __debug__:
     raise RuntimeError('Run verification without python -O; assertions are required.')
    p=data/'processed'
    report=json.loads((p/'smalltalk_multiturn.report.json').read_text())
    sp=spm.SentencePieceProcessor(model_file=str(data/'sp_korean.model'))
    text=(p/'smalltalk_multiturn.txt').read_text()
    manifest=[json.loads(s) for s in (p/'smalltalk_multiturn.manifest.jsonl').read_text().splitlines()]
    fragments=text.rstrip('\n').split('\n\n')
    assert len(fragments)==len(manifest)==report['output_fragments']
    tokens=turns=0
    sources=Counter(); transition_set=set(); windows=Counter(); emissions=defaultdict(list)
    for block,record in zip(fragments,manifest):
     lines=block.splitlines()
     assert len(lines)%2==0
     assert all(line.startswith('질문: ' if i%2==0 else '답변: ') for i,line in enumerate(lines))
     n=len(sp.encode(block+'\n\n'))
     assert n<=384 and n==record['tokens']
     assert len(lines)//2==record['turns']
     tokens+=n;turns+=len(lines)//2;sources[record['source_type']]+=1
     if record['source_type']=='synthetic_transition':
      assert 4<=record['turns']<=8
      transition_set.add(block)
     else:
      windows[record['source_file']]+=1
      pairs=[(lines[i][len('질문: '):], lines[i+1][len('답변: '):]) for i in range(0,len(lines),2)]
      emissions[record['source_file']].append((record,pairs))
    assert tokens==report['total_tokens'] and turns==report['output_turns']
    assert len(transition_set)==report['transition_sessions']==400
    assert hashlib.sha256((p/'smalltalk_multiturn.txt').read_bytes()).hexdigest()==report['output_sha256']
    assert gzip.decompress((p/'smalltalk_multiturn.txt.gz').read_bytes()).decode()==text
    assert hashlib.sha256((p/'qa_smalltalk_train.csv').read_bytes()).hexdigest()==report['old_csv_sha256_unchanged']
    assert hashlib.sha256((data/'sp_korean.model').read_bytes()).hexdigest()==report['tokenizer_sha256']
    schema=Counter(); digest=hashlib.sha256(); checked=0
    for path in sorted((data/'raw_aihub/라벨링데이터').rglob('*.json')):
     raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
     relative=path.relative_to(data/'raw_aihub/라벨링데이터').as_posix()
     digest.update(relative.encode()+b'\0'+sha.encode()+b'\n')
     obj=json.loads(raw.decode('utf-8-sig'))
     if relative in emissions:
      utterances=obj['utterances']
      ids=[u.get('utterance_id','') for u in utterances]
      if all(isinstance(i,str) and re.search(r'[.\-_](\d+)$',i) for i in ids):
       utterances=sorted(utterances,key=lambda u:int(re.search(r'(\d+)$',u['utterance_id']).group(1)))
      original=[]
      for index in range(0,len(utterances),2):
       q,a=utterances[index:index+2]
       assert q['speaker']=='speakerA' and a['speaker']=='speakerB' and q.get('turn_id')==a.get('turn_id')
       original.append((re.sub(r'\s+',' ',q['text']).strip(),re.sub(r'\s+',' ',a['text']).strip()))
      start=0
      for window_index,(record,output_pairs) in enumerate(emissions[relative]):
       assert record['source_sha256']==sha and record['window_index']==window_index
       end=start+len(output_pairs)
       assert output_pairs==original[start:end], relative
       start=end-1 if len(output_pairs)>=2 else end
      assert end==len(original), relative
     if 'utterances' in obj:schema['sns_utterances']+=1
     elif 'sessionInfo' in obj:schema['other_corpus_sessionInfo']+=1
     else:schema['unknown']+=1
     checked+=1
    assert digest.hexdigest()==report['input_manifest_sha256']
    result={'validated_fragments':len(fragments),'validated_turns':turns,'validated_tokens':tokens,
     'transition_sessions':len(transition_set),'token_overflow_count':0,'all_original_json_hashes_unchanged':True,
     'source_files_rehashed':checked,'output_qa_matches_source_utterances':True,
     'manifest_source_hashes_match':True,'output_sha256_matches_report':True,'source_schema_counts':dict(schema),
     'existing_csv_and_tokenizer_unchanged':True,'gzip_roundtrip_valid':True,
     'sns_sessions_emitted':len(windows),'text_bytes':(p/'smalltalk_multiturn.txt').stat().st_size,
     'compressed_bytes':(p/'smalltalk_multiturn.txt.gz').stat().st_size}
    if args.report:
     with args.report.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    else:
     print('Read-only verification; existing validation report left unchanged.')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
