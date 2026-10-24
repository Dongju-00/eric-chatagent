import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from model.scripts.prepare_multiturn_smalltalk import aligned_pairs, pair_windows, render_session


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(text)


class SmalltalkTests(unittest.TestCase):
    def test_numeric_order_and_pairing(self):
        utterances = []
        for n in range(1, 7):
            utterances += [
                {'speaker': 'speakerA', 'text': f'Q{n}', 'turn_id': f's-{n}', 'utterance_id': f's.{2*n-1}'},
                {'speaker': 'speakerB', 'text': f'A{n}', 'turn_id': f's-{n}', 'utterance_id': f's.{2*n}'},
            ]
        pairs, reason = aligned_pairs({'utterances': list(reversed(utterances))})
        self.assertIsNone(reason)
        self.assertEqual(pairs, [(f'Q{n}', f'A{n}') for n in range(1, 7)])

    def test_rejects_third_speaker_or_wrong_turn(self):
        items = [{'speaker':'speakerA','text':'Q','turn_id':'s-1'},
                 {'speaker':'speakerB','text':'A','turn_id':'s-2'}]
        self.assertEqual(aligned_pairs({'utterances': items})[1], 'turn_id_mismatch')
        items.append({'speaker':'speakerC','text':'C'})
        self.assertEqual(aligned_pairs({'utterances': items})[1], 'three_or_more_speakers')

    def test_unrelated_corpus_is_not_misreported_as_corrupt(self):
        self.assertEqual(aligned_pairs({'sessionInfo': []})[1], 'non_target_corpus_sessionInfo')

    def test_complete_pair_windows_and_overlap(self):
        pairs = [(f'Q{i}', f'A{i}') for i in range(8)]
        windows = pair_windows(pairs, CharacterTokenizer(), 45)
        self.assertGreater(len(windows), 1)
        self.assertEqual(set(pairs), {pair for window in windows for pair in window})
        for window in windows:
            text = render_session(window)
            self.assertTrue(text.startswith('질문: '))
            self.assertLessEqual(len(text), 45)
            self.assertEqual(len(text.splitlines()) % 2, 0)
        self.assertEqual(windows[0][-1], windows[1][0])

    def test_single_oversize_pair_rejected_without_truncation(self):
        with self.assertRaises(ValueError):
            pair_windows([('x' * 80, 'answer')], CharacterTokenizer(), 45)


if __name__ == '__main__':
    unittest.main()
