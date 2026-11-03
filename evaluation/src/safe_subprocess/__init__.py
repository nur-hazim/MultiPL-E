import os
import signal
import fcntl
import time
import subprocess
from typing import List

MAX_BYTES_PER_READ = 1024
SLEEP_BETWEEN_READS = 0.1


class Result:
    timeout: int
    exit_code: int
    stdout: str
    stderr: str

    def __init__(self, timeout, exit_code, stdout, stderr):
        self.timeout = timeout
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


def set_nonblocking(reader):
    fd = reader.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)


def run(
    args: List[str],
    timeout_seconds: int = 15,
    max_output_size: int = 2048,
    env = None,
    cwd: str | None = None
) -> Result:
    """
    Runs the given program with arguments. After the timeout elapses, kills the process
    and all other processes in the process group. Captures at most max_output_size bytes
    of stdout and stderr each, and discards any output beyond that.
    """
    p = subprocess.Popen(
        args,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        bufsize=MAX_BYTES_PER_READ,
        cwd=cwd
    )
    stdout_saved_bytes = bytearray()
    stderr_saved_bytes = bytearray()
    exit_code = None
    try:
        set_nonblocking(p.stdout)
        set_nonblocking(p.stderr)
        # start_new_session=True makes the process-group ID equal to p.pid.
        # Avoid getpgid(): a very short-lived child can exit before that lookup.
        max_iterations = max(1, int(timeout_seconds * 10))
        for _ in range(max_iterations):
            for reader, saved in ((p.stdout, stdout_saved_bytes),
                                  (p.stderr, stderr_saved_bytes)):
                chunk = reader.read(MAX_BYTES_PER_READ)
                if chunk:
                    saved.extend(chunk[:max(0, max_output_size - len(saved))])
            exit_code = p.poll()
            if exit_code is not None:
                break
            time.sleep(SLEEP_BETWEEN_READS)
    finally:
        try:
            # Kill descendants even when the direct child has exited normally.
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        finally:
            try:
                # Reap the direct child on success, timeout, and read/setup errors.
                p.wait()
            finally:
                p.stdout.close()
                p.stderr.close()

    timeout = exit_code is None
    exit_code = exit_code if exit_code is not None else -1
    stdout = bytes(stdout_saved_bytes).decode("utf-8", errors="ignore")
    stderr = bytes(stderr_saved_bytes).decode("utf-8", errors="ignore")
    return Result(timeout=timeout, exit_code=exit_code, stdout=stdout, stderr=stderr)
