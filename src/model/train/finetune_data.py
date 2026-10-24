"""Shared standalone data/loss helpers, copied into reviewed Colab bundles.

No checkpoint, network or training side effects on import. Model block_size stays
384; the training-sample limit may be smaller without changing parameter shapes.
"""
from collections import defaultdict
import random
from pathlib import Path
import os


def render_pairs(pairs):
    return ''.join(f'질문: {q}\n답변: {a}\n' for q,a in pairs)


def parse_pairs(text):
    lines=text.strip('\n').splitlines()
    if len(lines)%2:
        raise ValueError('Incomplete QA pair')
    pairs=[]
    for i in range(0,len(lines),2):
        if not lines[i].startswith('질문: ') or not lines[i+1].startswith('답변: '):
            raise ValueError('Unexpected role order')
        pairs.append((lines[i][4:],lines[i+1][4:]))
    return pairs


def window_pairs(pairs,sp,limit=256):
    if any(len(sp.encode(render_pairs([p])+'\n'))>limit for p in pairs):
        raise ValueError('Single QA pair exceeds the requested window; never truncate it')
    windows=[];start=0
    while start<len(pairs):
        end=start+1
        while end<len(pairs) and len(sp.encode(render_pairs(pairs[start:end+1])+'\n'))<=limit:
            end+=1
        windows.append(pairs[start:end])
        if end==len(pairs):break
        start=end-1 if end-start>=2 else end
    return windows


def split_groups(keys,val_fraction=.05,seed=20261024):
    """Stratify source families; every original source/template stays in one split."""
    families=defaultdict(list)
    for key in sorted(set(keys)):
        families[key.split(':',1)[0]].append(key)
    train,val=set(),set()
    rng=random.Random(seed)
    for family,group in sorted(families.items()):
        rng.shuffle(group)
        n=max(1,round(len(group)*val_fraction)) if len(group)>1 else 0
        val.update(group[:n]);train.update(group[n:])
    return train,val


def answer_mask(ids,sp):
    answer=sp.piece_to_id('답변:');newline=sp.piece_to_id('\n')
    if answer<=0 or newline<=0:raise ValueError('Dedicated answer/newline token missing')
    mask=[];line_start=True;in_answer=False
    for token in ids:
        if token==newline:
            mask.append(in_answer)
            line_start=True;in_answer=False
        elif line_start and sp.id_to_piece(token)=='▁':
            mask.append(False)
        elif line_start:
            in_answer=token==answer
            line_start=False;mask.append(False)
        else:
            mask.append(in_answer)
    return mask


def collate_samples(samples,pad_id=0,max_length=384):
    import torch
    length=max(len(s['input_ids']) for s in samples)
    if length>max_length:raise ValueError('Over-budget sample')
    xs=[];ys=[]
    for sample in samples:
        ids=sample['input_ids'];mask=sample['answer_mask']
        if len(ids)!=len(mask) or not any(mask[1:]):raise ValueError('No supervised answer target')
        pad=length-len(ids)
        xs.append(ids+[pad_id]*pad)
        ys.append([token if keep else -100 for token,keep in zip(ids,mask)]+[-100]*pad)
    return torch.tensor(xs,dtype=torch.long),torch.tensor(ys,dtype=torch.long)


def causal_loss(model,x,y,reduction='mean'):
    import torch.nn.functional as F
    logits,_=model(x)
    targets=y[:,1:].contiguous()
    if not targets.ne(-100).any():raise ValueError('No shifted supervision')
    return F.cross_entropy(logits[:,:-1,:].contiguous().view(-1,logits.shape[-1]),
                           targets.view(-1),ignore_index=-100,reduction=reduction)


def evaluate_token_weighted(model,loader,device):
    import torch
    was_training=model.training;model.eval()
    total=0.;count=0
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device);y=y.to(device)
            total+=causal_loss(model,x,y,reduction='sum').item()
            count+=int(y[:,1:].ne(-100).sum())
    model.train(was_training)
    if not count:raise ValueError('Empty validation supervision')
    return total/count


def save_torch_state(state,path):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    torch.save(state,temporary)
    os.replace(temporary,path)
