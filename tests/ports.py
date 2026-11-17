"""A port for a test that has to name one before the process that binds it exists."""

from __future__ import annotations

import itertools
import os
import socket

#: The ports handed out: below what Linux (from 32768) and macOS (from 49152) give to a bind to
#: port 0 by default, in blocks of ``_BLOCK`` so that every test process has ports of its own.
_FIRST = 21000
_BLOCKS = 110
_BLOCK = 100
_next = itertools.count()


def reserved_port() -> int:
    """A free port that the caller's child processes can bind later.

    Asking the system (bind to port 0, read the port, close) only says that the port was free at
    that moment. A process group of ``torch.distributed`` binds it seconds later, and until then
    any other bind to port 0 on the machine, of which a test run makes many, may be given the same
    port. The ports returned here lie outside the range such binds draw from (where that range is
    the system's default), and the block a process takes them from follows from its process id:
    the workers of one pytest-xdist run, whose ids are close together, never share a block.
    """
    start = _FIRST + (os.getpid() % _BLOCKS) * _BLOCK
    for _ in range(_BLOCK):
        port = start + next(_next) % _BLOCK
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:  # taken by something outside this process
                continue
        return port
    raise RuntimeError(f"no free port in {start}..{start + _BLOCK - 1}")
