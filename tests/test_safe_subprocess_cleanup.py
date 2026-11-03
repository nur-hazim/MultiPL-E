"""Regression tests for Linux pipe cleanup; mocks also run on Windows."""
import gc
import importlib.util
import io
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import warnings

SOURCE = Path(__file__).resolve().parents[1] / 'evaluation' / 'src' / 'safe_subprocess' / '__init__.py'


def load_runner():
    spec = importlib.util.spec_from_file_location('cleanup_test_runner', SOURCE)
    module = importlib.util.module_from_spec(spec)
    # The actual fcntl operations are mocked for lifecycle tests on Windows.
    if os.name == 'nt':
        with patch.dict(sys.modules, {'fcntl': SimpleNamespace()}):
            spec.loader.exec_module(module)
    else:
        spec.loader.exec_module(module)
    return module


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.module = load_runner()
        self.process = SimpleNamespace(pid=4321, stdout=io.BytesIO(b'output'),
                                       stderr=io.BytesIO(b'error'), poll=Mock(return_value=0),
                                       wait=Mock(return_value=0))
        self.killer = Mock()
        for target, name, value in [
            (self.module.subprocess, 'Popen', Mock(return_value=self.process)),
            (self.module, 'set_nonblocking', Mock()),
            (self.module.os, 'killpg', self.killer),
            (self.module, 'signal', SimpleNamespace(SIGKILL=9)),
            (self.module.time, 'sleep', Mock()),
        ]:
            patcher = patch.object(target, name, value, create=True)
            patcher.start()
            self.addCleanup(patcher.stop)

    def assert_cleaned(self):
        self.killer.assert_called_once_with(4321, 9)
        self.process.wait.assert_called_once()
        self.assertTrue(self.process.stdout.closed)
        self.assertTrue(self.process.stderr.closed)

    def test_success_closes_pipes_and_reaps_child(self):
        result = self.module.run(['candidate'], max_output_size=4)
        self.assertEqual((result.stdout, result.stderr, result.exit_code), ('outp', 'erro', 0))
        self.assertFalse(result.timeout)
        self.assert_cleaned()

    def test_timeout_closes_pipes_and_reaps_child(self):
        self.process.poll.return_value = None
        result = self.module.run(['candidate'], timeout_seconds=1)
        self.assertTrue(result.timeout)
        self.assertEqual(result.exit_code, -1)
        self.assert_cleaned()

    def test_setup_exception_still_cleans_up(self):
        self.module.set_nonblocking.side_effect = RuntimeError('setup failed')
        with self.assertRaisesRegex(RuntimeError, 'setup failed'):
            self.module.run(['candidate'])
        self.assert_cleaned()

    def test_read_exception_still_cleans_up(self):
        class BrokenReader(io.BytesIO):
            def read(self, size=-1):
                raise OSError('read failed')
        self.process.stdout.close()
        self.process.stdout = BrokenReader()
        with self.assertRaisesRegex(OSError, 'read failed'):
            self.module.run(['candidate'])
        self.assert_cleaned()

    def test_exited_group_still_closes_and_reaps(self):
        self.killer.side_effect = ProcessLookupError()
        self.assertEqual(self.module.run(['candidate']).exit_code, 0)
        self.assert_cleaned()

    def test_kill_exception_still_closes_and_reaps(self):
        self.killer.side_effect = PermissionError('kill failed')
        with self.assertRaisesRegex(PermissionError, 'kill failed'):
            self.module.run(['candidate'])
        self.assert_cleaned()


@unittest.skipUnless(sys.platform.startswith('linux'), 'Native Linux process tests')
class LinuxProcessTests(unittest.TestCase):
    def test_repeated_runs_do_not_leak_file_descriptors(self):
        module = load_runner()
        before = len(list(Path('/proc/self/fd').iterdir()))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ResourceWarning)
            for _ in range(10):
                result = module.run([sys.executable, '-c', 'print("finished", flush=True)'])
                self.assertEqual(result.exit_code, 0)
                self.assertEqual(result.stdout, 'finished\n')
            gc.collect()
        self.assertFalse([w for w in caught if issubclass(w.category, ResourceWarning)])
        self.assertEqual(len(list(Path('/proc/self/fd').iterdir())), before)

    def test_timeout_reaps_actual_child(self):
        module = load_runner()
        result = module.run([sys.executable, '-c',
                             'import os,time;print(os.getpid(),flush=True);time.sleep(60)'],
                            timeout_seconds=1)
        self.assertTrue(result.timeout)
        with self.assertRaises(ProcessLookupError):
            os.kill(int(result.stdout.strip()), 0)


if __name__ == '__main__':
    unittest.main()
