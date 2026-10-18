"""Mechanically regenerate the pinned final20 manifest; no PR/model calls."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reviewer.benchmark import generate_final_subset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', default='.cache/reviewbench/ReviewBench-968291d99e82b3e2969d7e0b4acca62585886c13.tar.gz')
    parser.add_argument('--dev-subset', default='configs/reviewbench-dev5.json')
    parser.add_argument('--output', default='configs/reviewbench-final20.json')
    args = parser.parse_args()
    raw = generate_final_subset(args.archive, args.dev_subset)
    Path(args.output).write_bytes(raw)
    from reviewer.benchmark import digest, read_json
    data = read_json(raw)
    print(f"tasks={len(data['tasks'])} sha256={digest(raw)}")
    for task in data['tasks']:
        print(task['task_id'])


if __name__ == '__main__':
    main()
