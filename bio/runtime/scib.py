"""Build scIB's bundled graph-LISI source against the current host ABI.

Only the executable lookup is redirected; official metric algorithms are unchanged.
No installed package or public_code file is modified.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import shutil
import subprocess
import threading
from unittest.mock import patch

from bio.io.paths import PROJECT_ROOT

_LOCK = threading.RLock()


@contextmanager
def compatible_lisi_binary():
    import scib
    package = Path(scib.__file__).resolve().parent
    source = package / 'knn_graph/knn_graph.cpp'
    compiler = shutil.which('g++')
    if not source.is_file() or compiler is None:
        yield  # Let the official bundled executable run, or report its failure.
        return
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    runtime = PROJECT_ROOT / 'output/runtime_cache/scib' / digest
    binary = runtime / 'knn_graph/knn_graph.o'
    with _LOCK:
        if not binary.is_file():
            binary.parent.mkdir(parents=True, exist_ok=True)
            import os
            temporary = binary.with_name(f'knn_graph.{os.getpid()}.tmp')
            try:
                subprocess.run([compiler, '-O3', '-std=c++11', str(source), '-o', str(temporary)],
                               check=True, capture_output=True, text=True)
                temporary.replace(binary)
            finally:
                temporary.unlink(missing_ok=True)
        # scIB 1.1.x locates its graph executable relative to scib.__file__.
        with patch.object(scib, '__file__', str(runtime / '__init__.py')):
            yield


def lisi(fn, *args, **kwargs):
    with compatible_lisi_binary():
        return fn(*args, **kwargs)
