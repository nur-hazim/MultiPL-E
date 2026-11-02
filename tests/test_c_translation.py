"""C translation coverage and executable oracle fixtures (not corpus solutions).

Run from the fork root: python -m unittest discover -s tests -p test_c_translation.py -v
Set CC to a C11 compiler. No model, GPU, datasets package or network is needed.
"""
import ast
from contextlib import redirect_stdout
from copy import deepcopy
import io
import os
import re
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dataset_builder'))
sys.path.insert(0, str(ROOT / 'evaluation' / 'src'))
from humaneval_to_c import Translator
from eval_c import eval_script
from prepare_humaneval_c import prepare
from evaluate_c_completions import evaluate


# HumanEval 151's reference body rejects all floats (including 5.0), but its
# supplied tests require integral-valued floats to contribute. The tests are
# authoritative by the corpus policy. These seven explicit test-only fixtures
# preserve their exact expected outputs without changing the original source.
TASK_151_FIXTURES = {
    (): 0,
    (5.0, 4.0): 25,
    (0.1, 0.2, 0.3): 0,
    (-10.0, -20.0, -30.0): 0,
    (-1.0, -2.0, 8.0): 0,
    (0.2, 3.0, 5.0): 34,
    (-9.0, -7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0): 165,
}


def translate(path, doctests='keep'):
    # Upstream generic_translator opens terms.csv relative to the current cwd.
    old = Path.cwd()
    try:
        os.chdir(ROOT / 'dataset_builder')
        from generic_translator import translate_prompt_and_tests
        tr = Translator()
        with redirect_stdout(io.StringIO()) as log:
            pair = translate_prompt_and_tests(path, tr, doctests, 'reworded')
        if pair is None:
            raise AssertionError(f'{path.name}: translation failed\n{log.getvalue()}')
        return tr, pair
    finally:
        os.chdir(old)


class TranslationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = os.environ.get('CC', 'gcc')
        if not shutil.which(cls.compiler):
            raise RuntimeError('C compiler required: install GCC or set CC. Tests must not silently skip.')

    def evaluate(self, program, timeout=15):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.c'
            path.write_text(program, encoding='utf-8')
            return eval_script(path, compiler=self.compiler, timeout_seconds=timeout)

    def test_all_originals_translate_and_execute(self):
        sources = sorted((ROOT / 'datasets' / 'originals-with-cleaned-doctests').glob('*.py'))
        self.assertEqual(len(sources), 161, 'Pinned HumanEval corpus changed; audit coverage before updating this count')
        total = 0
        for source in sources:
            with self.subTest(task=source.name):
                tr, (prompt, tests) = translate(source)
                namespace = {}
                exec(compile(source.read_text(encoding='utf-8'), str(source), 'exec'), namespace)
                if source.stem == 'HumanEval_151_double_the_difference':
                    self.assertEqual(namespace[tr.entry]([5.0, 4.0]), 0,
                                     'Reference discrepancy changed; review the explicit fixture exception')
                calls = []
                def capture(*args):
                    before = deepcopy(args)
                    if source.stem == 'HumanEval_151_double_the_difference':
                        # Unknown inputs fail with KeyError rather than guessing
                        # semantics outside the seven supplied test cases.
                        answer = TASK_151_FIXTURES[tuple(args[0])]
                    else:
                        answer = namespace[tr.entry](*args)
                    calls.append((before, deepcopy(answer)))
                    return answer
                namespace['check'](capture)
                self.assertGreater(len(calls), 0)
                # Every original assertion must survive translation.
                parsed = ast.parse(source.read_text(encoding='utf-8'))
                check = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == 'check')
                nasserts = sum(isinstance(n, ast.Assert) for n in ast.walk(check))
                self.assertEqual(tests.count('assert('), nasserts)
                self.assertGreaterEqual(len(calls), nasserts)
                total += nasserts
                # This test-only lookup fixture verifies actual arguments
                # against Python inputs and supplies expected outputs. It is
                # intentionally unsuitable as a HumanEval implementation.
                globals_, cases = [], []
                for index, (args, answer) in enumerate(calls):
                    name = f'he_fixture_answer_{index}'
                    literal = tr.literal(answer, tr.ret)
                    # Static C11 aggregates need brace initializers, rather
                    # than struct-valued compound expressions. Nested array
                    # compound literals have static duration at file scope.
                    literal = re.sub(r'\(\(he_\w+\)\{', '{', literal).replace('})', '}')
                    globals_.append(f'static {tr.ctype(tr.ret)} {name} = {literal};')
                    checks = [f'{tr.eqname(spec)}(he_arg_{i}, {tr.literal(arg, spec)})'
                              for i, (arg, spec) in enumerate(zip(args, tr.args))]
                    for i, (arg, spec) in enumerate(zip(args, tr.args)):
                        if spec == ('value',):
                            checks.append(f'he_arg_{i}.tag == ({tr.literal(arg,spec)}).tag')
                    cases.append('if (' + ' && '.join(checks) + f') return {name};')
                # Equality helpers precede the fixture's function body.
                helpers = '\n'.join(tr.test_suite_prefix_lines(tr.entry)[:-1])
                signature = prompt[prompt.rfind('\n' + tr.ctype(tr.ret) + ' he_') + 1:]
                fixture = (tr.header() + helpers + '\n' + '\n'.join(globals_) + '\n'
                           + signature + '\n' + '\n'.join(cases) + '\n_Exit(3);\n}\n')
                # Avoid repeating helper definitions already included above.
                main = tests[tests.index('int main(void) {'):]
                result = self.evaluate(fixture + main)
                self.assertEqual(result['status'], 'OK', result['stderr'])
        self.assertEqual(total, 1130)

    def test_evaluator_accepts_c_and_rejects_failures(self):
        programs = [
            ('#include <assert.h>\nint main(void){assert(2+2==4);}', 'OK'),
            ('#include <assert.h>\nint main(void){assert(0);}', 'Exception'),
            ('#include <vector>\nint main(void){}', 'SyntaxError'),
            ('int main(void){ this is invalid; }', 'SyntaxError'),
            ('int main(void){return 3;}', 'Exception'),
            ('int main(void){for(;;){}}', 'Timeout'),
        ]
        for program, status in programs:
            with self.subTest(status=status, program=program):
                self.assertEqual(self.evaluate(program, timeout=1 if status == 'Timeout' else 15)['status'], status)

    def test_int64_boundaries_and_utf8(self):
        tr = Translator()
        lo = tr.literal(-(2**63), ('int',))
        hi = tr.literal(2**63-1, ('int',))
        text = tr.literal('é9"\\\n', ('str',))
        result = self.evaluate(tr.header() + f'\nint main(void){{assert({lo}<0);assert({hi}>0);assert(strlen({text})==6);}}')
        self.assertEqual(result['status'], 'OK', result['stderr'])
        with self.assertRaises(ValueError):
            tr.literal(2**63, ('int',))
        with self.assertRaises(ValueError):
            tr.literal('a\0b', ('str',))

    def test_real_implementations_and_wrong_answers(self):
        source = ROOT / 'datasets' / 'originals-with-cleaned-doctests' / 'HumanEval_0_has_close_elements.py'
        tr, (prompt, tests) = translate(source)
        good = ('for(size_t i=0;i<he_arg_0.length;i++) for(size_t j=i+1;j<he_arg_0.length;j++) '
                'if(fabs(he_arg_0.data[i]-he_arg_0.data[j])<he_arg_1) return true; return false;\n}\n')
        self.assertEqual(self.evaluate(prompt + good + tests)['status'], 'OK')
        self.assertEqual(self.evaluate(prompt + 'return false;\n}\n' + tests)['status'], 'Exception')
        self.assertEqual(self.evaluate(prompt + 'return false;\n}\n#define NDEBUG\n' + tests)['status'], 'Exception')

    def test_runtime_numeric_tags_are_preserved(self):
        path = ROOT / 'datasets' / 'originals-with-cleaned-doctests' / 'HumanEval_92_any_int.py'
        tr, (prompt, tests) = translate(path)
        self.assertEqual(tr.args, [('value',)] * 3)
        good = ('if(he_arg_0.tag!=2 || he_arg_1.tag!=2 || he_arg_2.tag!=2) return false; '
                'int64_t x=he_arg_0.integer,y=he_arg_1.integer,z=he_arg_2.integer; '
                'return x+y==z || x+z==y || y+z==x;\n}\n')
        self.assertEqual(self.evaluate(prompt + good + tests)['status'], 'OK')

    def test_container_comparisons_detect_wrong_values(self):
        cases = [
            (('list', ('int',)), [1, 2], [2, 1]),
            (('tuple', ('str',), ('bool',)), ('a', True), ('a', False)),
            (('dict', ('str',), ('int',)), {'a': 1}, {'a': 2}),
            (('value',), None, 0),
            (('value',), [1, 'x'], [1, 'y']),
        ]
        for spec, a, b in cases:
            with self.subTest(spec=spec):
                tr = Translator()
                tr.ret = spec
                tr.ctype(spec)
                helpers = '\n'.join(tr.test_suite_prefix_lines('fixture')[:-1])
                program = tr.header() + helpers + f'\nint main(void){{assert(!{tr.eqname(spec)}({tr.literal(a,spec)},{tr.literal(b,spec)}));}}'
                result = self.evaluate(program)
                self.assertEqual(result['status'], 'OK', result['stderr'])

    def test_preparation_is_complete_and_fails_on_missing_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'prompts.jsonl'
            manifest = prepare(ROOT / 'datasets' / 'originals-with-cleaned-doctests', target)
            self.assertEqual(manifest['tasks'], 161)
            self.assertEqual(manifest['assertions'], 1130)
            self.assertEqual(len(target.read_text(encoding='utf-8').splitlines()), 161)
            missing = Path(tmp) / 'empty'
            missing.mkdir()
            with self.assertRaises(ValueError):
                prepare(missing, Path(tmp) / 'bad.jsonl')
            self.assertFalse((Path(tmp) / 'bad.jsonl').exists())

    def test_generation_run_validation_and_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            incoming = root / 'incoming'
            incoming.mkdir()
            rows = [dict(name=f'HumanEval_{i}_fixture', language='c', tests='/* frozen tests */')
                    for i in range(161)]
            expected = root / 'expected.jsonl'
            expected.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Missing generation'):
                evaluate(incoming, root / 'results', expected)
            for row in rows:
                (incoming / (row['name'] + '.json')).write_text(
                    json.dumps(dict(row, prompt='', completions=['/* candidate */'])), encoding='utf-8')
            with patch('evaluate_c_completions.eval_script', return_value=dict(status='OK', exit_code=0, stdout='', stderr='')):
                with redirect_stdout(io.StringIO()):
                    report = evaluate(incoming, root / 'results', expected, root / 'export')
            self.assertTrue(report['complete'])
            self.assertEqual(len(list((root / 'export').glob('*.c'))), 161)
            altered = dict(rows[0], tests='/* altered tests */', completions=[''])
            (incoming / (rows[0]['name'] + '.json')).write_text(json.dumps(altered), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Tests differ'):
                evaluate(incoming, root / 'results', expected)


if __name__ == '__main__':
    unittest.main()
