import sys
import contextlib
import io
import csv
import copy
import hashlib
import json
import tempfile
import zipfile
from types import SimpleNamespace
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from model.scripts.prepare_multiturn_news import normalize_title, existing_titles, validate_conversation, split_articles, write_dataset, clean_text, validate_dataset

class WordTokenizer:
    def encode(self, text, **kwargs):
        return text.split()

class NewsTests(unittest.TestCase):
    def candidate(self):
        result = {'context': '제목: 시험 기업의 발표\n내용: 새 서비스 계획을 발표했다. 세부 일정은 검토 중이다. 구체적인 운영 방안은 검토 후 안내할 예정이라고 밝혔다.',
                'turns': [
                 {'question': '어떤 발표야?', 'answer': '참고 뉴스 기준으로 새 서비스 계획을 발표했습니다.', 'evidence': '새 서비스 계획을 발표했다.', 'unanswerable': False},
                 {'question': '그건 어떤 상태야?', 'answer': '참고 뉴스 기준으로 세부 일정은 검토 중입니다.', 'evidence': '세부 일정은 검토 중이다.', 'unanswerable': False},
                 {'question': '방금 말한 일정은?', 'answer': '참고 뉴스에서는 검토 중이라고 설명했습니다.', 'evidence': '세부 일정은 검토 중이다.', 'unanswerable': False},
                 {'question': '그럼 확정 날짜는?', 'answer': '참고 뉴스에 해당 정보가 없습니다.', 'evidence': '', 'unanswerable': True},
                 {'question': '그 소식의 근거는?', 'answer': '참고 뉴스 기준으로 새 서비스 계획을 발표했습니다.', 'evidence': '새 서비스 계획을 발표했다.', 'unanswerable': False}]}

        for turn in result['turns']:
            if not turn['unanswerable']:
                turn['answer'] = f'뉴스에서는 "{turn["evidence"]}"라고 설명했습니다.'
        return result

    def test_valid_grounded_followup_conversation(self):
        source = {'body': self.candidate()['context'], 'title': '시험 기업의 발표'}
        result = validate_conversation(self.candidate(), source, WordTokenizer(), 1)
        self.assertEqual(result['turns'], 5)
        self.assertGreaterEqual(result['dependent_followups'], 2)

    def test_rejects_invented_number_and_unverifiable_evidence(self):
        c = self.candidate(); source={'body': c['context'], 'title': '시험 기업의 발표'}
        c['turns'][0]['answer'] += ' 매출은 999억원입니다.'
        with self.assertRaises(ValueError):validate_conversation(c, source, WordTokenizer(), 1)
        c=self.candidate();c['turns'][0]['evidence']='기록적인 이익'
        with self.assertRaises(ValueError):validate_conversation(c, source, WordTokenizer(), 1)

    def test_rejects_wrong_refusal_ratio_or_token_overflow(self):
        c=self.candidate();source={'body': c['context'], 'title': '시험 기업의 발표'}
        with self.assertRaises(ValueError):validate_conversation(c, source, WordTokenizer(), 2)
        with self.assertRaises(ValueError):validate_conversation(c, source, WordTokenizer(), 1,max_tokens=5)

    def test_rejects_recommendations_with_korean_particles_and_unrelated_claims(self):
        for answer in ['참고 뉴스 기준으로 지금 매수를 추천합니다.',
                       '뉴스에서는 매도를 권장합니다.',
                       '참고 뉴스 기준으로 그 회사는 파산했습니다.']:
            c = self.candidate(); source = {'title': '시험 기업의 발표', 'body': c['context']}
            c['turns'][0]['answer'] = answer
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                validate_conversation(c, source, WordTokenizer(), 1)

    def test_rejects_fabricated_context_even_if_evidence_is_real(self):
        c = self.candidate(); source = {'title': '시험 기업의 발표', 'body': c['context']}
        c['context'] += ' 기록적인 성장을 달성했다.'
        with self.assertRaises(ValueError):
            validate_conversation(c, source, WordTokenizer(), 1)
        with self.assertRaises(ValueError):
            validate_conversation([], source, WordTokenizer(), 1)

    def test_final_csv_is_article_disjoint_and_has_exact_28_percent_refusals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / 'original.csv'
            original.write_text('question,answer\n', encoding='utf-8')
            sources, conversations = [], {}
            for index in range(600):
                c = copy.deepcopy(self.candidate())
                c['context'] = c['context'].replace('시험 기업', f'시험 기업 {index}')
                if index % 5 < 2:
                    c['turns'][4] = {'question': '그럼 확정 시각은?',
                        'answer': '참고 뉴스에 해당 정보가 없습니다.', 'evidence': '', 'unanswerable': True}
                source = {'source_id': str(index), 'title': f'시험 기업 {index}의 발표', 'body': c['context']}
                source['body_sha256'] = hashlib.sha256(clean_text(source['body']).encode()).hexdigest()
                sources.append(source)
                conversations[str(index)] = c
            tokenizer_path = root/'sp_korean.model'
            tokenizer_path.write_bytes(b'fixture-tokenizer')
            args = SimpleNamespace(existing=original, output_dir=root/'output', work_dir=root/'cache', target=600, tokenizer=tokenizer_path)
            with contextlib.redirect_stdout(io.StringIO()):
                write_dataset(args, sources, conversations, WordTokenizer())
            report = json.loads((args.output_dir/'qa_news_multiturn.report.json').read_text())
            self.assertEqual(report['turns'], 3000)
            self.assertEqual(report['refusal_ratio'], .28)
            archive_path = args.output_dir/'news_multiturn_colab.zip'
            self.assertTrue(archive_path.is_file())
            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(archive.read('sp_korean.model'), b'fixture-tokenizer')
                self.assertNotIn('.env', archive.namelist())
                train_records = [json.loads(line) for line in archive.read('news_train.jsonl').decode().splitlines()]
                val_records = [json.loads(line) for line in archive.read('news_val.jsonl').decode().splitlines()]
                self.assertEqual((len(train_records), len(val_records)), (540,60))
                self.assertFalse({r['conversation_id'] for r in train_records} & {r['conversation_id'] for r in val_records})
                self.assertTrue(all(r['text'].startswith('참고 뉴스:\n') and r['text'].count('질문: ')==5 for r in train_records))
                self.assertTrue(all(r['token_count']<=384 for r in train_records))
                self.assertEqual(json.loads(archive.read('manifest.json'))['articles'],600)
            groups = []
            for suffix, expected in [('_train',2700),('_val',300)]:
                with (args.output_dir/f'qa_news_multiturn{suffix}.csv').open(encoding='utf-8-sig') as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), expected)
                self.assertTrue(all(bool(row['context']) == (row['turn_index']=='0') for row in rows))
                groups.append({row['conversation_id'] for row in rows})
            self.assertFalse(groups[0] & groups[1])
            with self.assertRaises(FileExistsError):
                write_dataset(args, sources, conversations, WordTokenizer())
            with self.assertRaises(ValueError):
                validate_dataset(args, sources[:-1], conversations, WordTokenizer())
            duplicate = [*sources[:-1], sources[0]]
            with self.assertRaises(ValueError):
                validate_dataset(args, duplicate, conversations, WordTokenizer())
            bad_hash = copy.deepcopy(sources)
            bad_hash[0]['body_sha256'] = 'tampered'
            with self.assertRaises(ValueError):
                validate_dataset(args, bad_hash, conversations, WordTokenizer())

    def test_split_disjoint_and_title_normalized(self):
        articles=[{'source_id':str(i),'title':str(i)} for i in range(600)]
        train,val=split_articles(articles)
        self.assertEqual((len(train),len(val)),(540,60))
        self.assertFalse({r['source_id'] for r in train}&{r['source_id'] for r in val})
        self.assertEqual(normalize_title(' <b>삼성</b> &amp;  뉴스 '),normalize_title('삼성 & 뉴스'))

if __name__=='__main__':unittest.main()
