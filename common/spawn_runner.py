import multiprocessing as mp
import os
import subprocess
from typing import Dict, List, Optional


def _child(argv: List[str], env_updates: Optional[Dict[str, str]] = None) -> None:
    """
    Spawned child entrypoint. Executes the provided argv via subprocess.run
    with parent env + env_updates, then exits with the same return code.
    """
    env = os.environ.copy()
    if env_updates:
        env.update(env_updates)
    rc = subprocess.run(argv, env=env).returncode
    os._exit(rc)  # propagate exact exit code to parent


def spawn(argv: List[str], env_updates: Optional[Dict[str, str]] = None, start: bool = True) -> mp.Process:
    """
    Non-blocking: start a spawned process that runs `subprocess.run(argv, env=...)`.
    `argv` should include the interpreter if needed, e.g.:
        [sys.executable, "/path/to/script.py", "--flag", "val"]
    Returns the multiprocessing.Process (caller should .join()).
    """
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_child, args=(list(argv), env_updates), daemon=False)
    if start:
        p.start()
    return p


def run(argv: List[str], env_updates: Optional[Dict[str, str]] = None) -> int:
    """
    Blocking convenience wrapper around spawn(...).
    Returns the child exit code.
    """
    p = spawn(argv, env_updates)
    p.join()
    return p.exitcode
