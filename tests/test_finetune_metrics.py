import sys,unittest,tempfile,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from model.train.finetune_metrics import build_evaluation_cases, audit_multiturn, summarize_rows, save_comparison, measure_model

class MetricsTests(unittest.TestCase):
    def record(self):
        return {'conversation_id':'sample','source_key':'heldout:1',
                'text':'참고 뉴스:\n제목: 예시\n내용: 회사는 창고를 점검했다.\n\n질문: 어떤 소식이야?\n답변: 창고를 점검했어.\n질문: 그럼 뭘 점검했어?\n답변: 창고를 점검했어.\n'}

    def test_cases_freeze_and_history_ablation_keep_current_question(self):
        a=build_evaluation_cases([self.record()],'news',limit=4)
        b=build_evaluation_cases([self.record()],'news',limit=4)
        self.assertEqual(a,b)
        case=next(c for c in a if c['category']=='multi_turn')
        self.assertIn('어떤 소식이야?',case['prompt'])
        self.assertNotIn('어떤 소식이야?',case['no_history_prompt'])
        self.assertTrue(case['prompt'].endswith('질문: 그럼 뭘 점검했어?\n답변: '))
        controlled=[c for c in a if c['category']=='controlled_reference']
        self.assertEqual(len(controlled),6)
        self.assertEqual(controlled[0]['question'],controlled[1]['question'])
        self.assertNotEqual(controlled[0]['expected_anchor'],controlled[1]['expected_anchor'])

    def test_audit_counts_later_answer_targets(self):
        audit=audit_multiturn([self.record()])
        self.assertEqual(audit['multiturn_samples'],1)
        self.assertEqual(audit['later_turns'],1)
        self.assertEqual(audit['samples'],1)

    def test_summary_separates_no_history_gain_and_anchor_proxy(self):
        rows=[{'category':'multi_turn','answer_nll':1.,'no_history_nll':2.,'answer_tokens':3,
               'generated_tokens':5,'latency_seconds':.5,'reference_f1':.8,'repetition_ratio':.1,
               'empty_answer':False,'expected_refusal':False,'predicted_refusal':False,
               'anchor_hit':None,'pair_id':None},
              {'category':'controlled_reference','answer_nll':2.,'no_history_nll':1.,'answer_tokens':1,
               'generated_tokens':5,'latency_seconds':.5,'reference_f1':.5,'repetition_ratio':0.,
               'empty_answer':False,'expected_refusal':False,'predicted_refusal':False,
               'anchor_hit':True,'pair_id':'p'}]
        result=summarize_rows(rows)
        self.assertEqual(result['answer_nll'],1.25)
        self.assertEqual(result['history_gain_nll'],.5)
        self.assertEqual(result['controlled_anchor_match_rate'],1.)
        self.assertIsNone(result['controlled_pair_match_rate'])

    def test_persisted_comparison_has_both_phases_and_explicit_deltas(self):
        before={'aggregate':{'answer_nll':2.},'rows':[],'protocol_sha256':'same'}
        after={'aggregate':{'answer_nll':1.},'rows':[],'protocol_sha256':'same'}
        with tempfile.TemporaryDirectory() as d:
            save_comparison(Path(d),before,after)
            result=json.loads((Path(d)/'comparison.json').read_text())
            self.assertEqual(result['delta_after_minus_before']['answer_nll'],-1.)
            self.assertTrue((Path(d)/'comparison.csv').exists())
            after['protocol_sha256']='different'
            with self.assertRaises(ValueError):save_comparison(Path(d),before,after)

    def test_measurement_supervises_only_last_answer_and_preserves_rng(self):
        import re,torch
        from model.train.finetune_data import answer_mask
        class TinyTokenizer:
            def __init__(self):
                self.pieces={0:'<pad>',1:'<unk>',3:'<eos>',4:'질문:',5:'답변:',6:'\n'}
                self.ids={v:k for k,v in self.pieces.items()}
            def encode(self,text,out_type=int):
                result=[]
                for token in re.findall(r'질문:|답변:|\n|.',text):
                    if token not in self.ids:
                        i=max(self.pieces)+1;self.pieces[i]=token;self.ids[token]=i
                    result.append(self.ids[token])
                return result
            def decode(self,ids):return ''.join(self.pieces[i] for i in ids)
            def piece_to_id(self,piece):return self.ids[piece]
            def id_to_piece(self,i):return self.pieces[i]
            def eos_id(self):return 3
        sp=TinyTokenizer();sp.encode('a')
        class TinyModel(torch.nn.Module):
            def forward(self,x):return torch.zeros((*x.shape,128)),None
            def generate(self,x,n,**kwargs):
                torch.rand(1)
                tail=torch.tensor([[sp.ids['a'],6]],dtype=torch.long)
                return torch.cat([x,tail],dim=1)
        model=TinyModel();model.train()
        case={'case_id':'probe','category':'multi_turn','source_key':'heldout:1','question':'q',
              'reference_answer':'a','prompt':'질문: x\n답변: zzzz\n질문: q\n답변: ',
              'no_history_prompt':'질문: q\n답변: ','context':'','expected_anchor':None,
              'foil_anchor':None,'pair_id':None,'expected_refusal':None}
        rng=torch.get_rng_state().clone()
        result=measure_model(model,sp,[case],'cpu',answer_mask,'smalltalk',1.,max_new_tokens=2)
        self.assertEqual(result['rows'][0]['answer_tokens'],3)  # separator space + answer + newline
        self.assertAlmostEqual(result['aggregate']['answer_nll'],__import__('math').log(128),places=5)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(model.training)

if __name__=='__main__':unittest.main()
