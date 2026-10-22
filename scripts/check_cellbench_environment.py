#!/usr/bin/env python3
"""Command entry point for bio.diagnostics.environment."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bio.diagnostics.environment import main


if __name__ == '__main__':
    raise SystemExit(main())
