#!/usr/bin/env python3
"""Export versioned metrics; do not combine native-scale errors with correlations."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bio.io.export import export_task, PRIMARY

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True,choices=list(PRIMARY))
    parser.add_argument('--root',default='output')
    args=parser.parse_args()
    export_task(args.task,args.root)
