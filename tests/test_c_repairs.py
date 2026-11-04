"""Repair regressions: preserve good candidates/tests; verify assembled C."""
from contextlib import redirect_stdout
import gzip
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dataset_builder'))
sys.path.insert(0, str(ROOT / 'evaluation' / 'src'))
from c_completion import normalize_c_source
from repair_c_completions import repair, ollama_repair, read_record
from eval_c import eval_script

HEADER = '#include <stdint.h>\n#include <stdbool.h>\n#include <assert.h>\ntypedef struct { int64_t *data; unsigned long length; } he_list;\n'
PREFIX = HEADER + 'he_list he_fixture(he_list he_arg_0) {\n'
FUNCTION = 'he_list he_fixture(he_list he_arg_0) { return he_arg_0; }\n'


class RepairTests(unittest.TestCase):
    def test_restore_only_omitted_declarations(self):
        source, changes = normalize_c_source(PREFIX, FUNCTION)
        self.assertEqual(source, HEADER + FUNCTION)
        self.assertEqual(changes, ['restored_frozen_preamble'])
        complete = HEADER + FUNCTION
        self.assertEqual(normalize_c_source(PREFIX, complete), (complete, []))

    def test_add_stdio_without_rewriting_function(self):
        body = HEADER + 'char *he_fixture(void) {static char b[30]; sprintf(b,"%d",3);return b;}'
        source, changes = normalize_c_source(PREFIX, body)
        self.assertEqual(source, '#include <stdio.h>\n' + body)
        self.assertEqual(changes, ['added_stdio_header'])

    def test_fences_and_invalid_extensions_are_not_guessed(self):
        code, changes = normalize_c_source(PREFIX, '```c\n' + FUNCTION.rstrip() + '\n```')
        self.assertEqual(code, HEADER + FUNCTION.rstrip())
        self.assertIn('removed_complete_markdown_fence', changes)
        broken = HEADER + FUNCTION + '```\nunfinished explanation'
        self.assertEqual(normalize_c_source(PREFIX, broken), (broken, []))
        extension = HEADER + 'he_list he_fixture(he_list a) { unsigned __int128 n=0; return a; }'
        self.assertEqual(normalize_c_source(PREFIX, extension), (extension, []))

    def test_recovered_declarations_compile_and_execute(self):
        compiler = os.environ.get('CC', 'gcc')
        if not shutil.which(compiler):
            raise RuntimeError('GCC required; set CC for repair regression tests')
        source, _ = normalize_c_source(PREFIX, FUNCTION)
        source += 'int main(void) {int64_t a[]={7};he_list x={a,1};assert(he_fixture(x).data[0]==7);}'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'candidate.c'
            path.write_text(source, encoding='utf-8')
            result = eval_script(path, compiler=compiler)
        self.assertEqual(result['status'], 'OK', result['stderr'])

    def make_run(self, root):
        incoming, results = root / 'originals', root / 'evaluated'
        incoming.mkdir()
        results.mkdir()
        expected = []
        snapshots = {}
        for i in range(161):
            name = f'HumanEval_{i}_fixture'
            row = dict(name=name, language='c', prompt=PREFIX, tests='/* frozen tests */')
            expected.append(row)
            source = FUNCTION if i < 13 else HEADER + FUNCTION
            problem = dict(row, prompt='', completions=[source])
            path = incoming / f'{name}.json.gz'
            with gzip.open(path, 'wt', encoding='utf-8') as stream:
                json.dump(problem, stream)
            snapshots[path.name] = path.read_bytes()
            record = dict(problem, results=[dict(status='SyntaxError' if i < 13 else 'OK',
                                                 exit_code=1 if i < 13 else 0,
                                                 stderr='unknown type' if i < 13 else '', stdout='',
                                                 program=source + '\n' + row['tests'])])
            del record['completions']
            (results / f'{name}.results.json').write_text(json.dumps(record), encoding='utf-8')
        frozen = root / 'expected.jsonl'
        frozen.write_text('\n'.join(map(json.dumps, expected)), encoding='utf-8')
        return incoming, results, frozen, snapshots

    def test_repair_calls_only_failed_tasks_and_preserves_originals(self):
        for model in (None, 'installed-ollama-model'):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                incoming, results, frozen, snapshots = self.make_run(root)
                output = root / 'new-run'
                with patch('repair_c_completions.ollama_repair', return_value=(FUNCTION, {'done': True}, {})) as request:
                    with redirect_stdout(io.StringIO()):
                        report = repair(incoming, results, frozen, output, model=model)
                self.assertEqual(request.call_count, 0 if model is None else 13)
                self.assertEqual((report['copied_passing'], report['appended_candidates']), (148, 13))
                for i in range(161):
                    name = f'HumanEval_{i}_fixture.json.gz'
                    self.assertEqual((incoming / name).read_bytes(), snapshots[name])
                    if i >= 13:
                        self.assertEqual((output / name).read_bytes(), snapshots[name])
                    else:
                        row = read_record(output / name)
                        self.assertEqual(row['completions'][0], FUNCTION)
                        self.assertEqual(row['completions'][1], HEADER + FUNCTION)
                        self.assertEqual(row['tests'], '/* frozen tests */')
                        self.assertEqual(row['repair_history'][0]['candidate_index'], 1)

    def test_stale_evaluations_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            incoming, results, frozen, _ = self.make_run(root)
            path = results / 'HumanEval_0_fixture.results.json'
            record = json.loads(path.read_text())
            record['results'][0]['program'] = 'different candidate'
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'does not match'):
                repair(incoming, results, frozen, root / 'new-run')
            self.assertFalse((root / 'new-run').exists())

    def test_ollama_response_protocol_and_truncation(self):
        for reason in ('stop', 'length'):
            response = dict(done=True, done_reason=reason, message=dict(content=FUNCTION))
            with patch('repair_c_completions.urlopen', return_value=io.BytesIO(json.dumps(response).encode())) as http:
                if reason == 'length':
                    with self.assertRaisesRegex(ValueError, 'incomplete'):
                        ollama_repair('model:tag', 'http://example:11434', PREFIX, FUNCTION, 'error', 9000, 32000, 60)
                else:
                    raw, _, options = ollama_repair('model:tag', 'http://example:11434', PREFIX, FUNCTION, 'error', 9000, 32000, 60)
                    self.assertEqual(raw, FUNCTION)
                    self.assertEqual(options['num_predict'], 9000)
                    request = http.call_args.args[0]
                    self.assertEqual(request.full_url, 'http://example:11434/api/chat')
                    self.assertFalse(json.loads(request.data)['stream'])


if __name__ == '__main__':
    unittest.main()
