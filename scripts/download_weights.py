#!/usr/bin/env python3
"""Command entry point for bio.assets.weights."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bio.assets.weights import main


if __name__ == '__main__':
    raise SystemExit(main())
