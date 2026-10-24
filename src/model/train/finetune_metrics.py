"""Reproducible before/best evaluation; standalone source embedded in Colab.

History-ablation NLL and lexical anchor/F1 scores are diagnostic proxies, not
proof of semantic understanding. Actual model metrics are measured only when
Colab executes this code; local data audits never fabricate model performance.
"""
import csv
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import re
import time

REFERENCE_PATTERN=re.compile(r'그\s*(?:회사|거|건|럼|내용|소식)|거기|방금|아까|앞서|더 자세히')
REFUSAL_PATTERN=re.compile(r'정보가 없습니다|확인할 수 없습니다|알 수 없습니다|뉴스에 없|제공되지')
PAIR_PATTERN=re.compile(r'^질문: ([^\n]*)\n답변: ([^\n]*)(?:\n|$)',re.MULTILINE)


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def protocol_sha256(cases):
    return hashlib.sha256(json.dumps(cases,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def unpack_turns(text):
    matches=list(PAIR_PATTERN.finditer(text))
    if not matches:raise ValueError('No complete QA turns')
    return text[:matches[0].start()],[(m.group(1),m.group(2)) for m in matches]


def audit_multiturn(records):
    counts=Counter();histogram=Counter()
    for r in records:
        _,pairs=unpack_turns(r['text']);n=len(pairs)
        counts['samples']+=1;histogram[str(n)]+=1
        counts['turns']+=n;counts['later_turns']+=max(0,n-1)
        counts['multiturn_samples']+=n>=2
        counts['reference_later_questions']+=sum(bool(REFERENCE_PATTERN.search(q)) for q,a in pairs[1:])
    return {**counts,'turn_count_histogram':dict(histogram),
            'multiturn_sample_ratio':counts['multiturn_samples']/max(1,counts['samples']),
            'later_reference_expression_ratio':counts['reference_later_questions']/max(1,counts['later_turns']),
            'caveat':'Reference-expression matches are lexical signals, not semantic correctness.'}


def controlled_cases(task):
    cases=[]
    if task=='smalltalk':
        for index,(a,b) in enumerate([('등산','수영'),('독서','요리'),('사진 촬영','자전거')]):
            for side,(target,foil) in enumerate([(a,b),(b,a)]):
                q='방금 내가 했다고 말한 활동은 뭐였지?'
                history=f"질문: 나 오늘 '{target}' 활동을 했어.\n답변: 오늘 {target} 활동을 했구나. 어땠어?\n"
                cases.append({'case_id':f'controlled-{index}-{side}','category':'controlled_reference',
                    'source_key':'controlled_synthetic:'+str(index),'question':q,'reference_answer':target,
                    'prompt':history+f'질문: {q}\n답변: ','no_history_prompt':f'질문: {q}\n답변: ',
                    'context':'','expected_anchor':target,'foil_anchor':foil,'pair_id':str(index),'expected_refusal':None})
    else:
        for index,(a,b) in enumerate([('포장 설비','물류 창고'),('연구 시설','생산 장비'),('배송 차량','보안 시스템')]):
            context=f'제목: 점검 안내(평가용 가상 자료)\n내용: A기업은 {a}를 점검했다. B기업은 {b}를 점검했다.'
            head='참고 뉴스:\n'+context+'\n\n';q='그 회사가 점검한 대상은 뭐야?'
            for side,(company,target,foil) in enumerate([('A기업',a,b),('B기업',b,a)]):
                particle='을' if (ord(target[-1])-0xAC00)%28 else '를'
                history=f'질문: {company}의 소식은?\n답변: {company}은 {target}{particle} 점검했습니다.\n'
                cases.append({'case_id':f'controlled-{index}-{side}','category':'controlled_reference',
                    'source_key':'controlled_synthetic:'+str(index),'question':q,'reference_answer':target,
                    'prompt':head+history+f'질문: {q}\n답변: ','no_history_prompt':head+f'질문: {q}\n답변: ',
                    'context':context,'expected_anchor':target,'foil_anchor':foil,'pair_id':str(index),'expected_refusal':False})
    return cases


def build_evaluation_cases(records,task,limit=16,seed=20261024):
    candidates=defaultdict(list)
    for r in sorted(records,key=lambda r:r['conversation_id']):
        head,pairs=unpack_turns(r['text'])
        for index,(q,a) in enumerate(pairs):
            if index==0 and REFERENCE_PATTERN.search(q):continue
            if index>0 and not REFERENCE_PATTERN.search(q):continue
            category='single_turn' if index==0 else 'multi_turn'
            current=f'질문: {q}\n답변: '
            history=''.join(f'질문: {x}\n답변: {y}\n' for x,y in pairs[:index])
            candidates[category].append({'case_id':r['conversation_id']+':turn'+str(index),
                'category':category,'source_key':r['source_key'],'question':q,'reference_answer':a,
                'prompt':head+history+current,'no_history_prompt':head+current if index else None,
                'context':head,'expected_anchor':None,'foil_anchor':None,'pair_id':None,
                'expected_refusal':bool(REFUSAL_PATTERN.search(a)) if task=='news' else None})
    rng=random.Random(seed);result=[]
    for category in ['single_turn','multi_turn']:
        pool=candidates[category];rng.shuffle(pool);chosen=[];used=set()
        # Include both news refusal and factual follow-ups in the frozen probe set.
        if task=='news' and category=='multi_turn':
            for c in pool:
                if c['expected_refusal'] and c['source_key'] not in used:
                    chosen.append(c);used.add(c['source_key'])
                    if len(chosen)>=max(1,round(limit*.28)):break
        for c in pool:
            if len(chosen)>=limit:break
            if c['source_key'] in used:continue
            if task=='news' and category=='multi_turn' and c['expected_refusal']:continue
            chosen.append(c);used.add(c['source_key'])
        result.extend(chosen)
    return result+controlled_cases(task)


def lexical_f1(answer,reference):
    a=re.sub(r'\s+','',answer);b=re.sub(r'\s+','',reference)
    aa=Counter(a[i:i+2] for i in range(max(1,len(a)-1))) if a else Counter()
    bb=Counter(b[i:i+2] for i in range(max(1,len(b)-1))) if b else Counter()
    common=sum((aa&bb).values())
    return 2*common/(sum(aa.values())+sum(bb.values())) if aa and bb else 0.


def repetition_ratio(answer):
    tokens=answer.split()
    grams=[tuple(tokens[i:i+4]) for i in range(max(0,len(tokens)-3))]
    return (len(grams)-len(set(grams)))/len(grams) if grams else 0.


def last_answer_nll(model,sp,prompt,answer,device,mask_fn):
    import torch
    import torch.nn.functional as F
    ids=sp.encode(prompt+answer+'\n',out_type=int)
    if len(ids)>384:raise ValueError('Evaluation target exceeds model window')
    mask=mask_fn(ids,sp)
    end=max(i for i,keep in enumerate(mask) if keep)
    start=end
    while start>0 and mask[start-1]:start-=1
    labels=[token if start<=i<=end else -100 for i,token in enumerate(ids)]
    x=torch.tensor([ids],device=device);y=torch.tensor([labels],device=device)
    with torch.no_grad():
        logits,_=model(x)
        count=int(y[:,1:].ne(-100).sum())
        loss=F.cross_entropy(logits[:,:-1,:].reshape(-1,logits.shape[-1]),y[:,1:].reshape(-1),ignore_index=-100,reduction='sum')
    return float(loss.item()/count),count


def percentile(values,fraction):
    if not values:return None
    ordered=sorted(values);position=(len(values)-1)*fraction;lo=int(position);hi=math.ceil(position)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(position-lo)


def summarize_rows(rows):
    def weighted(items,key):
        good=[r for r in items if r.get(key) is not None]
        n=sum(r['answer_tokens'] for r in good)
        return sum(r[key]*r['answer_tokens'] for r in good)/n if n else None
    nll=weighted(rows,'answer_nll')
    history=[r for r in rows if r.get('no_history_nll') is not None]
    gain=sum((r['no_history_nll']-r['answer_nll'])*r['answer_tokens'] for r in history)/sum(r['answer_tokens'] for r in history) if history else None
    anchors=[r['anchor_hit'] for r in rows if r.get('anchor_hit') is not None]
    pairs=defaultdict(list)
    for r in rows:
        if r.get('pair_id') is not None:pairs[r['pair_id']].append(r['anchor_hit'])
    complete_pairs=[values for values in pairs.values() if len(values)==2]
    positives=[r for r in rows if r.get('expected_refusal') is True]
    negatives=[r for r in rows if r.get('expected_refusal') is False]
    seconds=sum(r['latency_seconds'] for r in rows)
    result={'cases':len(rows),'answer_nll':nll,'answer_perplexity':math.exp(nll) if nll is not None and nll<700 else None,
        'single_turn_answer_nll':weighted([r for r in rows if r['category']=='single_turn'],'answer_nll'),
        'multi_turn_answer_nll':weighted([r for r in rows if r['category']=='multi_turn'],'answer_nll'),
        'history_gain_nll':gain,
        'history_helped_case_rate':sum(r['answer_nll']<r['no_history_nll'] for r in history)/len(history) if history else None,
        'controlled_anchor_match_rate':sum(anchors)/len(anchors) if anchors else None,
        'controlled_pair_match_rate':sum(all(v) for v in complete_pairs)/len(complete_pairs) if complete_pairs else None,
        'reference_character_bigram_f1':sum(r['reference_f1'] for r in rows)/len(rows) if rows else None,
        'repeated_4gram_ratio':sum(r['repetition_ratio'] for r in rows)/len(rows) if rows else None,
        'empty_answer_rate':sum(r['empty_answer'] for r in rows)/len(rows) if rows else None,
        'refusal_recall_proxy':sum(r['predicted_refusal'] for r in positives)/len(positives) if positives else None,
        'false_refusal_rate_proxy':sum(r['predicted_refusal'] for r in negatives)/len(negatives) if negatives else None,
        'generation_latency_p50_seconds':percentile([r['latency_seconds'] for r in rows],.5),
        'generation_latency_p95_seconds':percentile([r['latency_seconds'] for r in rows],.95),
        'generated_tokens_per_second':sum(r['generated_tokens'] for r in rows)/seconds if seconds else None}
    for category in ['multi_turn','controlled_reference']:
        selected=[r for r in history if r['category']==category]
        result[category+'_history_gain_nll']=sum((r['no_history_nll']-r['answer_nll'])*r['answer_tokens'] for r in selected)/sum(r['answer_tokens'] for r in selected) if selected else None
    return result


def measure_model(model,sp,cases,device,mask_fn,task,validation_nll,max_new_tokens=80,seed=20261024):
    import torch
    gpu=str(device).startswith('cuda')
    def sync():
        if gpu:torch.cuda.synchronize()
    stop={sp.piece_to_id('\n')}
    if sp.eos_id()>=0:stop.add(sp.eos_id())
    rows=[];was_training=model.training;model.eval()
    devices=[torch.cuda.current_device()] if gpu else []
    with torch.random.fork_rng(devices=devices),torch.no_grad():
        # Warm up generation separately so initialization is not counted as latency.
        if cases:
            warm=torch.tensor([sp.encode(cases[0]['prompt'])],device=device)
            model.generate(warm,2,stop_tokens=stop,temperature=1.,top_k=1,repetition_penalty=1.3);sync()
        for index,case in enumerate(cases):
            torch.manual_seed(seed+index)
            nll,count=last_answer_nll(model,sp,case['prompt'],case['reference_answer'],device,mask_fn)
            ablated=None
            if case['no_history_prompt'] is not None:
                ablated,_=last_answer_nll(model,sp,case['no_history_prompt'],case['reference_answer'],device,mask_fn)
            ids=sp.encode(case['prompt'],out_type=int)
            x=torch.tensor([ids],device=device);sync();started=time.perf_counter()
            generated=model.generate(x,max_new_tokens,stop_tokens=stop,temperature=1.,top_k=1,repetition_penalty=1.3)
            sync();elapsed=time.perf_counter()-started
            output=generated[0,len(ids):].tolist();answer=sp.decode(output).strip()
            anchor=case['expected_anchor'];foil=case['foil_anchor']
            compact=lambda value:re.sub(r'\s+','',value)
            hit=(compact(anchor) in compact(answer) and compact(foil) not in compact(answer)) if anchor is not None else None
            rows.append({**case,'generated_answer':answer,'answer_nll':nll,'answer_tokens':count,
                'no_history_nll':ablated,'generated_tokens':len(output),'latency_seconds':elapsed,
                'reference_f1':lexical_f1(answer,case['reference_answer']),'repetition_ratio':repetition_ratio(answer),
                'empty_answer':not bool(answer),'predicted_refusal':bool(REFUSAL_PATTERN.search(answer)),'anchor_hit':hit})
    model.train(was_training)
    aggregate=summarize_rows(rows)
    aggregate.update(validation_answer_nll=validation_nll,validation_answer_perplexity=math.exp(validation_nll) if validation_nll<700 else None)
    return {'task':task,'measured_at':datetime.now(timezone.utc).isoformat(),'protocol_sha256':protocol_sha256(cases),
            'generation':{'top_k':1,'temperature':1.,'repetition_penalty':1.3,'max_new_tokens':max_new_tokens,'seed':seed},
            'environment':{'torch':str(torch.__version__),'cuda':torch.version.cuda,'device':str(device),
                'gpu':torch.cuda.get_device_name() if gpu else None,'platform':platform.platform()},
            'aggregate':aggregate,'rows':rows,
            'limitations':['Character F1, refusal strings and anchor matches are lexical proxies, not semantic accuracy.',
                'History-ablation gain measures likelihood sensitivity, not proof of correct reference resolution.',
                'Controlled cases are synthetic evaluation fixtures only; they are never training data.',
                'Generation latency is a warm batch-1 probe, not an end-to-end serving benchmark.']}


def save_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def save_comparison(folder,before,after):
    folder=Path(folder)
    if before['protocol_sha256']!=after['protocol_sha256']:raise ValueError('Evaluation protocols differ')
    if [r['case_id'] for r in before['rows']]!=[r['case_id'] for r in after['rows']]:raise ValueError('Evaluation case order differs')
    if before.get('generation')!=after.get('generation'):raise ValueError('Generation configurations differ')
    old_env,new_env=before.get('environment',{}),after.get('environment',{})
    timing_comparable=bool(old_env and new_env) and all(old_env.get(k)==new_env.get(k) for k in ['torch','cuda','device','gpu'])
    keys=sorted(set(before['aggregate'])|set(after['aggregate']))
    delta={key:after['aggregate'][key]-before['aggregate'][key] if isinstance(before['aggregate'].get(key),(int,float)) and isinstance(after['aggregate'].get(key),(int,float)) else None for key in keys}
    result={'protocol_sha256':before['protocol_sha256'],'before':before['aggregate'],'after':after['aggregate'],
            'delta_after_minus_before':delta,'semantic_accuracy':None,'manual_review_required':True,'timing_environment_matches':timing_comparable,
            'note':'Positive history_gain_nll means history lowers target NLL; all lexical scores require semantic review.'}
    save_json(folder/'comparison.json',result)
    with (folder/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['metric','before','after','delta_after_minus_before']);writer.writeheader()
        for key in keys:writer.writerow({'metric':key,'before':before['aggregate'].get(key),'after':after['aggregate'].get(key),'delta_after_minus_before':delta[key]})
    fields=['case_id','category','source_key','prompt','reference_answer','before_answer','after_answer','before_semantic_pass','after_semantic_pass']
    with (folder/'manual_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for old,new in zip(before['rows'],after['rows']):
            writer.writerow({**{key:old[key] for key in fields[:5]},'before_answer':old['generated_answer'],
                             'after_answer':new['generated_answer'],'before_semantic_pass':'','after_semantic_pass':''})
    return result
