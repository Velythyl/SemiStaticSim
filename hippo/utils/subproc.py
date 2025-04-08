import dataclasses
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from io import StringIO
import queue

@dataclasses.dataclass
class SubprocessResult:
    returncode: int
    stdout: str
    stderr: str
    stdout_stderr: str

    @property
    def success(self):
        return self.returncode == 0

    @property
    def failure(self):
        return not self.success


def run_subproc(cmd, callback=None, shell=False):
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=shell,
        text=True,
        universal_newlines=True,
        bufsize=1  # line-buffered
    )

    stdout_buf = StringIO()
    stderr_buf = StringIO()
    combined_buf = StringIO()
    print_queue = queue.Queue()
    callback_queue = queue.Queue()
    lock = threading.Lock()

    if callback is None:
        class NullContext:
            def __enter__(self):
                pass  # Do nothing on enter

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass  # Do nothing on exit
        callback_lock = NullContext()
    else:
        callback_lock = threading.Lock()

    def reader(stream, buf, label):
        for line in iter(stream.readline, ''):
            #with callback_lock:
            buf.write(line)
            with lock:
                combined_buf.write(line)
            print_queue.put((label, line))
            callback_queue.put((label, line))
        stream.close()

    class STOPPRINT:
        pass

    def printer():
        while True:
            line = print_queue.get()

            if line is None:
                time.sleep(0.01)
                continue
            elif isinstance(line, STOPPRINT):
                break
            else:
                print(line[1], end='', file=sys.stderr if line[0] == 'stderr' else sys.stdout)

    def callbacker():
        while True:
            line = callback_queue.get()
            if line is None:
                time.sleep(0.01)
                continue
            elif isinstance(line, STOPPRINT):
                break
            else:
                partial_result = SubprocessResult(
                    returncode=None,
                    stdout=combined_buf.getvalue(),
                    stderr=stderr_buf.getvalue(),
                    stdout_stderr=combined_buf.getvalue(),
                )
                callback(partial_result)

    stdout_thread = threading.Thread(target=reader, args=(process.stdout, stdout_buf, 'stdout'))
    stderr_thread = threading.Thread(target=reader, args=(process.stderr, stderr_buf, 'stderr'))
    printer_thread = threading.Thread(target=printer)

    stdout_thread.start()
    stderr_thread.start()
    printer_thread.start()
    if callback is not None:
        callback_thread = threading.Thread(target=callbacker)
        callback_thread.start()

    process.wait()
    stdout_thread.join()
    stderr_thread.join()

    # Signal printer to stop
    print_queue.put(STOPPRINT())
    printer_thread.join()
    if callback is not None:
        callback_thread.put(STOPPRINT())
        callback_thread.join()

    result = SubprocessResult(
        returncode=process.returncode,
        stdout=stdout_buf.getvalue(),
        stderr=stderr_buf.getvalue(),
        stdout_stderr=combined_buf.getvalue()
    )

    if callback:
        callback(result)

    return result

if __name__ == "__main__":
    run_subproc("which python && ejcka", lambda result: print(result), shell=True)
