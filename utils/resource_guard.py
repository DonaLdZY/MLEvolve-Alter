from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply inherited POSIX resource limits before exec.")
    parser.add_argument("--memory-bytes", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("missing command after --")

    import resource

    memory_bytes = max(1, int(args.memory_bytes))
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    os.execvpe(command[0], command, os.environ.copy())
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
