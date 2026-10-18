"""Add lost scope labels to pinned ReviewBench detail JSON; matcher/scorer untouched.

Run against a writable extracted checkout of the exact pinned revision before
judging. This changes only detail serialization, preserving classifier output.
"""
import argparse
import hashlib
from pathlib import Path

PINNED_SOURCE_SHA256 = 'dd688b50ec14c3ae9239ec6850a1874e68b0657ca27be4de82066abb02bfd835'
PATCHED_SOURCE_SHA256 = '1eba4789458a83b313bb76fca267a563f77d72c93222cf095691602e530de99e'

ANCHORS = [
    ('  golden_category?: string;\n', '  golden_category?: string;\n  golden_scope?: string;\n'),
    ('  classifier_category?: string;\n', '  classifier_category?: string;\n  classifier_scope?: string;\n'),
    ('        golden_category: g.category,\n', '        golden_category: g.category,\n        golden_scope: g.scope,\n'),
    ('        classifier_category: cls.category,\n', '        classifier_category: cls.category,\n        classifier_scope: cls.scope,\n'),
]


def patch(path):
    path = Path(path)
    source = path.read_text(encoding='utf-8')
    if 'classifier_scope?: string;' in source:
        if (all(new in source for _, new in ANCHORS)
                and hashlib.sha256(source.encode()).hexdigest() == PATCHED_SOURCE_SHA256):
            return False
        raise ValueError('source is partially patched or differs from pinned ReviewBench')
    if hashlib.sha256(source.encode()).hexdigest() != PINNED_SOURCE_SHA256:
        raise ValueError('source hash differs from pinned ReviewBench revision')
    for old, new in ANCHORS:
        if source.count(old) != 1:
            raise ValueError('source does not match pinned ReviewBench detail serializer')
        source = source.replace(old, new, 1)
    path.write_text(source, encoding='utf-8')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reviewbench_root', help='Writable extraction of pinned ReviewBench revision')
    args = parser.parse_args()
    path = Path(args.reviewbench_root) / 'scripts/eval/eval.ts'
    changed = patch(path)
    print('patched' if changed else 'already patched')


if __name__ == '__main__':
    main()
