"""Preserve passing C tasks and append repairs only for failed tasks.

Without --ollama-model, only restore omitted frozen declarations/stdio.h. With
--ollama-model, request one repaired candidate per failed task from Ollama.
Always writes a NEW run; original candidates and frozen tests remain unchanged.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import Request, urlopen
from c_completion import normalize_c_source


def read_record(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def ollama_repair(model, api_base, prefix, source, diagnostic, max_tokens, num_ctx, timeout):
    request = dict(model=model, stream=False, messages=[
        dict(role='system', content=(
            'Repair a C implementation. Return only a COMPLETE ISO C11 source file. '
            'Preserve all supplied typedefs, struct layouts, names and parameter types. '
            'Put helper definitions/prototypes at FILE SCOPE, never inside a function. '
            'No nested functions, auto functions, __int128, C++, GNU extensions, tests, '
            'Markdown or commentary. Include required standard headers. '
            'Close every function/block. Return arrays/strings in heap or static storage. '
            'Implement the described algorithm for arbitrary inputs, not test-specific answers.')),
        dict(role='user', content=f'FROZEN INTERFACE AND TASK:\n{prefix}\n\nPREVIOUS SOURCE:\n{source}\n\nCOMPILER/TEST DIAGNOSTICS:\n{diagnostic}\n\nReturn the repaired complete C11 source.')
    ], options=dict(temperature=0.2, num_predict=max_tokens, num_ctx=num_ctx))
    req = Request(api_base.rstrip('/') + '/api/chat', data=json.dumps(request).encode('utf-8'),
                  headers={'Content-Type': 'application/json'}, method='POST')
    with urlopen(req, timeout=timeout) as response:
        result = json.load(response)
    if not result.get('done') or result.get('done_reason') == 'length':
        raise ValueError('Ollama response was incomplete/token-limited; increase the token/context limits')
    content = result.get('message', {}).get('content', '')
    if not content.strip():
        raise ValueError('Ollama returned no source code')
    return content, result, request['options']


def repair(generations, results, expected_path, output_dir, *, model=None,
           api_base='http://localhost:11434', max_tokens=16384, num_ctx=32768, timeout=600):
    generations, results, output_dir = map(Path, (generations, results, output_dir))
    rows = [json.loads(line) for line in Path(expected_path).read_text(encoding='utf-8').splitlines() if line.strip()]
    expected = {row['name']: row for row in rows}
    if len(rows) != 161 or len(expected) != 161:
        raise ValueError('Expected 161 unique frozen HumanEval tasks')
    inputs = {}
    for path in generations.iterdir():
        if not (path.name.endswith('.json') or path.name.endswith('.json.gz')) or '.results.' in path.name:
            continue
        row = read_record(path)
        name = row['name']
        if name in inputs or name not in expected:
            raise ValueError(f'Duplicate or unexpected task: {name}')
        if row.get('language') != 'c' or row.get('tests') != expected[name]['tests']:
            raise ValueError(f'Language/frozen tests mismatch for {name}')
        result_files = [p for p in (results / f'{name}.results.json.gz', results / f'{name}.results.json') if p.exists()]
        if len(result_files) != 1:
            raise ValueError(f'Expected exactly one evaluation record for {name}')
        evaluated = read_record(result_files[0])
        verdicts = evaluated['results']
        sources = [row.get('prompt', '') + c for c in row.get('completions', [])]
        if not sources or len(verdicts) != len(sources) or evaluated.get('tests') != row['tests']:
            raise ValueError(f'Stale or mismatched evaluation for {name}')
        for source, verdict in zip(sources, verdicts):
            if verdict.get('program') != source + '\n' + row['tests']:
                raise ValueError(f'Evaluation does not match the current candidate source: {name}')
        inputs[name] = (path, row, sources, verdicts)
    if set(inputs) != set(expected):
        raise ValueError('Generation run does not contain every frozen task')
    if output_dir.resolve() == generations.resolve() or output_dir.resolve() == results.resolve():
        raise ValueError('Use a new output directory, not the input/evaluation directory')
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('Output directory must be empty; original runs are never overwritten')
    output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(copied_passing=0, failed_tasks=0, appended_candidates=0, errors=[],
                  mode='ollama' if model else 'normalize', model=model,
                  prompts_sha256=hashlib.sha256(Path(expected_path).read_bytes()).hexdigest())
    for name, (path, row, sources, verdicts) in sorted(inputs.items()):
        destination = output_dir / path.name
        if any(v['status'] == 'OK' and v['exit_code'] == 0 for v in verdicts):
            shutil.copy2(path, destination)
            report['copied_passing'] += 1
            continue
        report['failed_tasks'] += 1
        # Preserve the original file even if the repair request fails.
        shutil.copy2(path, destination)
        try:
            raw = sources[-1]
            response, options = None, None
            if model:
                verdict = verdicts[-1]
                diagnostics = f'{verdict["status"]}\n{verdict.get("stderr", "")}\n{verdict.get("stdout", "")}'
                raw, response, options = ollama_repair(model, api_base, expected[name]['prompt'],
                                                       raw, diagnostics, max_tokens, num_ctx, timeout)
            code, changes = normalize_c_source(expected[name]['prompt'], raw)
            if not model and code == sources[-1]:
                print(f'{name}: no declaration-only repair available', flush=True)
                continue
            index = len(sources)
            row['prompt'] = ''
            row['completions'] = sources + [code]
            row.setdefault('repair_history', []).append(dict(
                candidate_index=index, normalization=changes, model=model,
                api_base=api_base if model else None, options=options,
                raw_completion=raw, response=response,
                source_sha256=hashlib.sha256(code.encode('utf-8')).hexdigest()))
            opener = gzip.open if destination.suffix == '.gz' else open
            with opener(destination, 'wt', encoding='utf-8') as stream:
                json.dump(row, stream)
            report['appended_candidates'] += 1
            print(f'{name}: appended candidate {index}; evaluation still required', flush=True)
        except Exception as exc:
            report['errors'].append(dict(task=name, error=str(exc)))
            print(f'{name}: repair failed: {exc}', flush=True)
    (output_dir / 'repair-summary.txt').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generations', required=True, type=Path)
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--expected', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--ollama-model', help='Exact Ollama model tag, without a LiteLLM provider prefix')
    parser.add_argument('--api-base', default='http://localhost:11434')
    parser.add_argument('--max-tokens', type=int, default=16384)
    parser.add_argument('--num-ctx', type=int, default=32768)
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    report = repair(args.generations, args.results, args.expected, args.output_dir,
                    model=args.ollama_model, api_base=args.api_base,
                    max_tokens=args.max_tokens, num_ctx=args.num_ctx, timeout=args.timeout)
    print(json.dumps(report, indent=2))
    if report['errors']:
        raise SystemExit(1)
