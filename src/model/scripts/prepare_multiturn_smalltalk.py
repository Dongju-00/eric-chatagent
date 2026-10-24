"""Convert AIHub sessions without flattening or overwriting existing datasets.

Run: uv run python src/model/scripts/prepare_multiturn_smalltalk.py
All outputs are new files. No model or vector-store imports are performed.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import time

import sentencepiece as spm

try:
    from _bootstrap import ensure_src_on_path
except ImportError:
    from model.scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()
from model.paths import DATA_DIR, SP_PREFIX  # noqa: E402


def normalize_text(value):
    if not isinstance(value, str):
        return ''
    return re.sub(r'\s+', ' ', value).strip()


def aligned_pairs(session):
    """Return complete A/B pairs, sorted by numeric utterance ID when available."""
    if 'utterances' not in session and 'sessionInfo' in session:
        return [], 'non_target_corpus_sessionInfo'
    utterances = session.get('utterances')
    if not isinstance(utterances, list) or not utterances:
        return [], 'missing_utterances'
    if not all(isinstance(u, dict) for u in utterances):
        return [], 'invalid_utterance'
    speakers = {u.get('speaker') for u in utterances}
    if len(speakers) >= 3:
        return [], 'three_or_more_speakers'
    if speakers != {'speakerA', 'speakerB'}:
        return [], 'invalid_speakers'
    ids = [u.get('utterance_id', '') for u in utterances]
    if all(isinstance(i, str) and re.search(r'[.\-_](\d+)$', i) for i in ids):
        if len(ids) != len(set(ids)):
            return [], 'duplicate_utterance_id'
        utterances = sorted(utterances, key=lambda u: int(re.search(r'(\d+)$', u['utterance_id']).group(1)))
    if len(utterances) % 2:
        return [], 'unpaired_utterance'
    pairs = []
    for i in range(0, len(utterances), 2):
        a, b = utterances[i:i + 2]
        if a.get('speaker') != 'speakerA' or b.get('speaker') != 'speakerB':
            return [], 'nonalternating_speakers'
        if a.get('turn_id') != b.get('turn_id'):
            return [], 'turn_id_mismatch'
        q, answer = normalize_text(a.get('text')), normalize_text(b.get('text'))
        if not q or not answer:
            return [], 'empty_text'
        pairs.append((q, answer))
    return pairs, None


def render_session(pairs):
    return ''.join(f'질문: {q}\n답변: {a}\n' for q, a in pairs)


def pair_windows(pairs, tokenizer, max_tokens=384):
    """Greedy complete-QA windows with one-pair overlap and guaranteed progress."""
    def fits(window):
        return len(tokenizer.encode(render_session(window) + '\n')) <= max_tokens
    if any(not fits([pair]) for pair in pairs):
        raise ValueError('single_pair_exceeds_token_limit')
    windows = []
    start = 0
    while start < len(pairs):
        end = start + 1
        while end < len(pairs) and fits(pairs[start:end + 1]):
            end += 1
        windows.append(pairs[start:end])
        if end == len(pairs):
            break
        start = end - 1 if end - start >= 2 else end
    return windows


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def convert(input_dir, output_dir, tokenizer_path, old_csv, transition_count=400):
    from model.scripts._transition_dialogues import transition_sessions

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    paths = sorted(input_dir.rglob('*.json'))
    if not paths:
        raise FileNotFoundError(f'No source JSON files in {input_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    names = ['smalltalk_multiturn.txt', 'smalltalk_multiturn.manifest.jsonl',
             'smalltalk_multiturn.report.json', 'smalltalk_multiturn.txt.gz',
             'smalltalk_transitions.txt']
    targets = [output_dir / name for name in names]
    if any(p.exists() for p in targets):
        raise FileExistsError('Output already exists; use a new --output-dir. Nothing overwritten.')
    old_hash = sha256_file(old_csv) if old_csv.exists() else None
    tokenizer_hash = sha256_file(tokenizer_path)
    old_rows = []
    if old_csv.exists():
        with old_csv.open(encoding='utf-8-sig', newline='') as f:
            old_rows = list(csv.DictReader(f))
    wanted = {normalize_text(row.get('question')) for row in old_rows}
    source_answers = defaultdict(set)
    rejected = Counter()
    stats = Counter()
    max_tokens = 0
    digest = hashlib.sha256()
    started = time.monotonic()
    # Every source read is hashed; only text plus a lightweight provenance record is emitted.
    with ExitStack() as stack:
        text = stack.enter_context(targets[0].open('x', encoding='utf-8', newline='\n'))
        manifest = stack.enter_context(targets[1].open('x', encoding='utf-8', newline='\n'))
        transitions = stack.enter_context(targets[4].open('x', encoding='utf-8', newline='\n'))

        def emit(pairs, identity, source_type, **metadata):
            nonlocal max_tokens
            serialized = render_session(pairs) + '\n'
            n = len(sp.encode(serialized))
            if n > 384:
                raise ValueError(f'Token budget violated by {identity}: {n}')
            text.write(serialized)
            if source_type == 'synthetic_transition':
                transitions.write(serialized)
            record = {'index': stats['output_fragments'], 'session_id': identity,
                      'source_type': source_type, 'turns': len(pairs), 'tokens': n, **metadata}
            manifest.write(json.dumps(record, ensure_ascii=False) + '\n')
            stats['output_fragments'] += 1
            stats['output_turns'] += len(pairs)
            stats['total_tokens'] += n
            max_tokens = max(max_tokens, n)

        for index, path in enumerate(paths):
            raw = path.read_bytes()
            relative = path.relative_to(input_dir).as_posix()
            source_hash = hashlib.sha256(raw).hexdigest()
            digest.update(relative.encode() + b'\0' + source_hash.encode() + b'\n')
            try:
                session = json.loads(raw.decode('utf-8-sig'))
                if not isinstance(session, dict):
                    raise ValueError('not_object')
                pairs, reason = aligned_pairs(session)
            except (ValueError, UnicodeError, TypeError):
                rejected['parse_failure'] += 1
                continue
            if reason:
                rejected[reason] += 1
                continue
            for q, a in pairs:
                if q in wanted:
                    source_answers[q].add(a)
            try:
                windows = pair_windows(pairs, sp)
            except ValueError:
                rejected['single_pair_exceeds_token_limit'] += 1
                continue
            stats['converted_source_sessions'] += 1
            stats['source_turns'] += len(pairs)
            if len(windows) > 1:
                stats['split_source_sessions'] += 1
                stats['fragments_from_split_sessions'] += len(windows)
            identity = str(session.get('info', {}).get('id') or relative)
            for window_index, window in enumerate(windows):
                emit(window, identity, 'aihub', source_file=relative,
                     source_sha256=source_hash, window_index=window_index)
            if (index + 1) % 5000 == 0:
                print(f'AIHub {index + 1}/{len(paths)}; converted={stats["converted_source_sessions"]}; fragments={stats["output_fragments"]}', flush=True)
        stats['aihub_output_fragments'] = stats['output_fragments']
        stats['aihub_output_turns'] = stats['output_turns']
        synthetic = transition_sessions(transition_count)
        for index, pairs in enumerate(synthetic):
            if not 4 <= len(pairs) <= 8:
                raise ValueError('Transition needs 4–8 turns')
            if len(pair_windows(pairs, sp)) != 1:
                raise ValueError('Transition session exceeds 384 tokens')
            emit(pairs, f'transition-{index:04d}', 'synthetic_transition', template_index=index // 20)
        stats['transition_sessions'] = len(synthetic)
        stats['transition_turns'] = sum(map(len, synthetic))

    mismatch_examples = []
    alignment = Counter()
    for row in old_rows:
        q, a = normalize_text(row.get('question')), normalize_text(row.get('answer'))
        originals = source_answers.get(q)
        if not originals:
            alignment['question_not_found_in_accepted_two_speaker_source'] += 1
        elif a in originals:
            alignment['exact_adjacent_pair_match'] += 1
        else:
            alignment['question_found_but_answer_differs_candidate'] += 1
            if len(mismatch_examples) < 15:
                mismatch_examples.append({'question': q, 'old_answer': a,
                                          'original_adjacent_answers': sorted(originals)[:3]})
    assert old_hash == (sha256_file(old_csv) if old_csv.exists() else None)
    assert tokenizer_hash == sha256_file(tokenizer_path)
    with targets[0].open('rb') as source, targets[3].open('xb') as output:
        with gzip.GzipFile(fileobj=output, mode='wb', mtime=0) as compressed:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                compressed.write(chunk)
    report = {'input_sessions': len(paths), **stats, 'rejected_sessions': sum(rejected.values()),
              'rejections_by_reason': dict(rejected), 'max_serialized_tokens': max_tokens,
              'over_budget_fragments': 0, 'input_manifest_sha256': digest.hexdigest(),
              'tokenizer_sha256': tokenizer_hash, 'old_csv_sha256_unchanged': old_hash,
              'old_csv_rows': len(old_rows), 'old_pair_comparison': dict(alignment),
              'candidate_mismatch_examples': mismatch_examples,
              'comparison_caveat': 'Different adjacent answers are alignment candidates, not proven semantic errors; unmatched questions cannot be classified. Rewritten valid answers can differ.',
              'output_bytes': targets[0].stat().st_size, 'gzip_bytes': targets[3].stat().st_size,
              'output_sha256': sha256_file(targets[0]), 'elapsed_seconds': round(time.monotonic() - started, 2)}
    with targets[2].open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'candidate_mismatch_examples'}, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=DATA_DIR / 'raw_aihub' / '라벨링데이터')
    parser.add_argument('--output-dir', type=Path, default=DATA_DIR / 'processed')
    parser.add_argument('--tokenizer', type=Path, default=Path(str(SP_PREFIX) + '.model'))
    parser.add_argument('--old-csv', type=Path, default=DATA_DIR / 'processed' / 'qa_smalltalk_train.csv')
    parser.add_argument('--transitions', type=int, default=400)
    args = parser.parse_args()
    convert(args.input_dir, args.output_dir, args.tokenizer, args.old_csv, args.transitions)


if __name__ == '__main__':
    main()
