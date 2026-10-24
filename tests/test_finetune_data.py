import sys
import unittest
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from model.train.finetune_data import window_pairs, split_groups, answer_mask, collate_samples, evaluate_token_weighted

class Tokenizer:
    def encode(self,text,**kwargs):return list(text)
    def piece_to_id(self,piece):return {'답변:':5,'질문:':4,'\n':6}[piece]
    def id_to_piece(self,i):return {5:'답변:',4:'질문:',6:'\n',7:'▁'}.get(i,'글')

class FinetuneDataTests(unittest.TestCase):
    def test_short_windows_preserve_complete_pairs(self):
        pairs=[('Q'+str(i),'A'+str(i)) for i in range(12)]
        windows=window_pairs(pairs,Tokenizer(),limit=44)
        self.assertEqual(set(pairs),{pair for window in windows for pair in window})
        self.assertTrue(all(len(window)>=1 for window in windows))

    def test_split_keeps_all_variants_of_source_together(self):
        keys=['aihub:'+str(i) for i in range(100)]+['synthetic:'+str(i) for i in range(20)]
        train,val=split_groups(keys,.05)
        self.assertFalse(train & val)
        self.assertEqual(train|val,set(keys))
        self.assertEqual(len(val),6)

    def test_marker_inside_context_is_not_an_answer(self):
        ids=[8,5,9,6,4,9,6,7,5,10,11,6]
        mask=answer_mask(ids,Tokenizer())
        self.assertEqual([i for i,x in enumerate(mask) if x],[9,10,11])

    def test_collate_is_dynamic_and_masks_padding_and_questions(self):
        a={'input_ids':[4,9,6,5,10,6],'answer_mask':[False,False,False,False,True,True]}
        b={'input_ids':[4,9,6,5,10,11,6],'answer_mask':[False,False,False,False,True,True,True]}
        x,y=collate_samples([a,b],pad_id=0,max_length=256)
        self.assertEqual(tuple(x.shape),(2,7))
        self.assertEqual(y[0,-1].item(),-100)
        self.assertTrue((y[:,:4]==-100).all())

    def test_validation_weights_tokens_and_restores_model_mode(self):
        import torch
        model=torch.nn.Linear(1,1)
        model.train()
        loader=[(torch.zeros((1,2),dtype=torch.long),torch.tensor([[-100,1]])),
                (torch.zeros((1,4),dtype=torch.long),torch.tensor([[-100,1,1,1]]))]
        with patch('model.train.finetune_data.causal_loss',side_effect=[torch.tensor(2.),torch.tensor(8.)]):
            loss=evaluate_token_weighted(model,loader,'cpu')
        self.assertEqual(loss,2.5)
        self.assertTrue(model.training)

if __name__=='__main__':unittest.main()
