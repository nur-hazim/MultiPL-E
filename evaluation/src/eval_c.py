"""Compile and execute C11 HumanEval programs, preserving assertions."""
import os
from pathlib import Path
import tempfile


def eval_script(path: Path, *, timeout_seconds=15, compiler='gcc'):
    # Use upstream process-group cleanup and bounded output on Linux. A small
    # Windows fallback supports local development; Ubuntu uses safe_subprocess.
    if os.name == 'nt':
        import subprocess
        def run(args, timeout_seconds=15):
            from types import SimpleNamespace
            try:
                p = subprocess.run(args, capture_output=True, text=True,
                                   errors='replace', timeout=timeout_seconds)
                return SimpleNamespace(timeout=False, exit_code=p.returncode,
                                       stdout=p.stdout[-2048:], stderr=p.stderr[-2048:])
            except subprocess.TimeoutExpired:
                return SimpleNamespace(timeout=True, exit_code=-1, stdout='', stderr='timeout')
    else:
        from safe_subprocess import run
    def result(p, failure):
        return dict(status='Timeout' if p.timeout else 'OK' if p.exit_code == 0 else failure,
                    exit_code=p.exit_code, stdout=p.stdout, stderr=p.stderr)
    with tempfile.TemporaryDirectory(prefix='multipl-e-c-') as tmp:
        binary = str(Path(tmp) / ('candidate.exe' if os.name == 'nt' else 'candidate'))
        extra = []
        if os.name == 'nt':
            # MinGW's CRT assertion can open a modal dialog and hang QA even
            # after terminating the candidate. Match assert's pass/fail and
            # NDEBUG behavior using a noninteractive header for Windows QA.
            # Ubuntu uses its system assert.h and the upstream process runner.
            header = Path(tmp) / 'assert.h'
            header.write_text('#include <stdlib.h>\n#include <stdio.h>\n#undef assert\n'
                              '#ifdef NDEBUG\n#define assert(e) ((void)0)\n#else\n'
                              '#define assert(e) ((e)?(void)0:(fprintf(stderr,"assertion failed\\n"),_Exit(3)))\n'
                              '#endif\n')
            extra = ['-I', str(header.parent)]
        try:
            built = run([compiler, '-x', 'c', '-std=c11', '-pedantic-errors',
                         '-O0', '-UNDEBUG', *extra, str(Path(path).resolve()), '-o', binary, '-lm'],
                        timeout_seconds=60)
        except OSError as exc:
            return dict(status='CompilerError', exit_code=-1, stdout='', stderr=str(exc))
        if built.timeout or built.exit_code != 0:
            return result(built, 'SyntaxError')
        # Generated programs are untrusted: execute this evaluator in the
        # project's evaluation container, not on a shared host with secrets.
        return result(run([binary], timeout_seconds=timeout_seconds), 'Exception')


if __name__ == '__main__':
    from generic_eval import main
    main(eval_script, 'C', '.c')
