"""Evaluate complete HumanEval C generation runs against a frozen prompt file.

Run in the C evaluation container. Writes upstream-compatible results files and,
optionally, exports the first passing candidate per task as a separate C corpus.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import tempfile
from eval_c import eval_script


def read_json(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def evaluate(directory, output, expected_path, export=None):
    with Path(expected_path).open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    expected = {row['name']: row for row in rows}
    if len(expected) != 161 or len(rows) != len(expected):
        raise ValueError('Expected 161 unique tasks in the frozen C prompt file')
    files = [p for p in Path(directory).iterdir()
             if (p.name.endswith('.json.gz') or p.name.endswith('.json'))
             and '.results.' not in p.name]
    problems = {}
    for path in files:
        problem = read_json(path)
        name = problem['name']
        if not re.fullmatch(r'HumanEval_\d+_[A-Za-z_][A-Za-z_0-9]*', name):
            raise ValueError(f'Invalid task name: {name!r}')
        if name in problems:
            raise ValueError(f'Duplicate generation file for {name}')
        if name not in expected or problem.get('language') != 'c':
            raise ValueError(f'Unexpected task/language: {name}')
        if problem.get('tests') != expected[name]['tests']:
            raise ValueError(f'Tests differ from frozen corpus for {name}')
        if not problem.get('completions'):
            raise ValueError(f'No generated candidates for {name}')
        problems[name] = problem
    if set(problems) != set(expected):
        raise ValueError(f'Missing generation tasks: {sorted(set(expected)-set(problems))}')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if export:
        export = Path(export)
        export.mkdir(parents=True, exist_ok=True)
        # Avoid presenting stale accepted files as part of the current run.
        if any(export.iterdir()):
            raise ValueError('Export directory must be empty; choose a new run directory')
    accepted = []
    for name in sorted(problems):
        problem = problems[name]
        results = []
        selected = None
        for index, completion in enumerate(problem['completions']):
            source = problem.get('prompt', '') + completion
            program = source + '\n' + expected[name]['tests']
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'candidate.c'
                path.write_text(program, encoding='utf-8')
                verdict = eval_script(path)
            results.append(dict(verdict, program=program))
            if selected is None and verdict['status'] == 'OK':
                selected = dict(name=name, candidate_index=index,
                                source_sha256=hashlib.sha256(source.encode('utf-8')).hexdigest())
                if export:
                    (export / f'{name}.c').write_text(source, encoding='utf-8')
                    (export / f'{name}_tests.c.txt').write_text(expected[name]['tests'], encoding='utf-8')
        record = {k: v for k, v in problem.items() if k != 'completions'}
        record['results'] = results
        with gzip.open(output / f'{name}.results.json.gz', 'wt', encoding='utf-8') as stream:
            json.dump(record, stream)
        if selected:
            accepted.append(selected)
        print(f'{name}: {sum(r["status"] == "OK" for r in results)}/{len(results)} candidates pass', flush=True)
    manifest = dict(tasks=161, tasks_with_passing_candidate=len(accepted), accepted=accepted,
                    prompts_sha256=hashlib.sha256(Path(expected_path).read_bytes()).hexdigest(),
                    complete=len(accepted) == 161)
    (output / 'corpus-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    if export:
        (export / 'corpus-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--expected', required=True, type=Path)
    parser.add_argument('--export-dir', type=Path)
    parser.add_argument('--require-all-pass', action='store_true')
    args = parser.parse_args()
    manifest = evaluate(args.dir, args.output_dir, args.expected, args.export_dir)
    print(f'{manifest["tasks_with_passing_candidate"]}/161 tasks have a passing C candidate')
    if args.require_all_pass and not manifest['complete']:
        raise SystemExit(1)
