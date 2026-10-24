"""Notebook contracts without fetching weights or starting training."""
import ast
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from model.scripts.prepare_colab_finetune import news_conversation

ROOT=Path(__file__).resolve().parents[1]

class ColabNotebookTests(unittest.TestCase):
    def test_both_notebooks_are_parseable_and_keep_drive_training_contract(self):
        for task in ['news','smalltalk']:
            path=ROOT/f'notebooks/colab/eric_{task}_multiturn_colab.ipynb'
            notebook=json.loads(path.read_text())
            sources=[]
            for cell in notebook['cells']:
                if cell['cell_type']!='code':continue
                source=''.join(cell['source'])
                if source.startswith('!'):continue
                ast.parse(source)
                sources.append(source)
                self.assertEqual(cell.get('outputs'),[])
            all_code='\n'.join(sources)
            self.assertIn("MODELS_DIR = DRIVE_ROOT / 'models'",all_code)
            self.assertIn("'max_epochs': 2",all_code)
            self.assertIn("'patience': 1",all_code)
            self.assertIn('evaluate_token_weighted',all_code)
            self.assertIn("BEFORE_PATH = RUN_DIR / 'before_metrics.json'",all_code)
            self.assertIn("AFTER_PATH = RUN_DIR / 'after_metrics.json'",all_code)
            self.assertIn("save_comparison(RUN_DIR, before_metrics, after_metrics)",all_code)
            self.assertIn('no_history_prompt',all_code)
            self.assertIn('dataset_audit.json',all_code)
            self.assertIn('model.load_state_dict(torch.load(CKPT_OUT',all_code)
            self.assertIn("LAST_STATE = RUN_DIR / 'last.resume.pt'",all_code)
            self.assertNotIn('qa_news_multiturn_free',all_code)
            self.assertNotIn('turns.pop()',all_code)
            self.assertNotIn('drop_last=True',all_code)
            stop_calls=[node for node in ast.walk(ast.parse(all_code))
                        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                        and node.func.attr=='piece_to_id']
            self.assertTrue(any(node.args[0].value=='\n' for node in stop_calls))

if __name__=='__main__':unittest.main()
