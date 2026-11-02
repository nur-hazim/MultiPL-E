"""Prepare the entire pinned HumanEval C corpus, failing on any skipped test.

Standard-library-only alternative to the HF builder. Python doctest examples
are retained as comments; executable unit tests are translated to C11.
"""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile


def prepare(originals, output):
    originals, output = Path(originals).resolve(), Path(output).resolve()
    sources = sorted(originals.glob('HumanEval_*.py'), key=lambda p: int(p.name.split('_')[1]))
    if len(sources) != 161:
        raise ValueError(f'Expected the 161 pinned HumanEval tasks, found {len(sources)}')
    old = Path.cwd()
    problems, hashes, assertions = [], {}, 0
    try:
        os.chdir(Path(__file__).resolve().parent)
        from generic_translator import translate_prompt_and_tests
        from humaneval_to_c import Translator
        import ast
        for source in sources:
            tr = Translator()
            with redirect_stdout(io.StringIO()) as log, redirect_stderr(log):
                pair = translate_prompt_and_tests(source, tr, 'keep', 'reworded')
            if pair is None:
                raise ValueError(f'{source.name}: C translation failed\n{log.getvalue()}')
            prompt, tests = pair
            tree = ast.parse(source.read_text(encoding='utf-8'))
            check = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'check')
            count = sum(isinstance(n, ast.Assert) for n in ast.walk(check))
            if not count or tests.count('assert(') != count:
                raise ValueError(f'{source.name}: translated assertion count mismatch')
            assertions += count
            hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
            problems.append(dict(name=source.stem, language='c', prompt=prompt, tests=tests,
                                 stop_tokens=[], doctests='keep', prompt_terminology='reworded',
                                 original=source.name))
    finally:
        os.chdir(old)
    if assertions != 1130:
        raise ValueError(f'Expected 1130 assertions, got {assertions}; audit source changes')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Do not publish a truncated JSONL if preparation fails.
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=output.parent,
                                     delete=False, suffix='.jsonl.tmp') as stream:
        temporary = Path(stream.name)
        for problem in problems:
            stream.write(json.dumps(problem, ensure_ascii=True) + '\n')
    os.replace(temporary, output)
    manifest = dict(tasks=len(problems), assertions=assertions, source_sha256=hashes,
                    jsonl_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    reference_validation_exceptions=[{
                        'task': 'HumanEval_151_double_the_difference',
                        'policy': 'Supplied tests are authoritative; seven explicit test-only expected-output fixtures replace the inconsistent Python reference during translator validation.',
                        'example': {'input': [5.0, 4.0], 'python_reference': 0, 'supplied_test': 25},
                    }],
                    interface='C11, signed int64, binary64, UTF-8 NUL strings, pointer/length containers')
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--originals', type=Path, default=Path(__file__).resolve().parents[1] / 'datasets' / 'originals-with-cleaned-doctests')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.originals, args.output)
    print(f'Prepared {result["tasks"]} C prompts with {result["assertions"]} assertions: {args.output}')
