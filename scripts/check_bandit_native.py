"""Offline native Bandit diagnostic for a known fixture and saved frozen PR input.

Run from the project root:
  python scripts/check_bandit_native.py \
    --run results/<run-id>/tools.json --snapshot-dir .cache/github/<pr-cache-dir>

This never calls a model, GitHub, or ReviewBench.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reviewer.archives import sha256
from reviewer.config import load_config
from reviewer.diff import parse_diff
from reviewer.github import Snapshot
from reviewer.retrieval import eligible, snapshot_sources
from reviewer.schemas import BaselineRun, ReviewInput
from reviewer.static_analysis import analyze_snapshot, get_bandit_version, run_bandit


def check_fixture(version):
    head, archive_hash = 'a' * 40, 'b' * 64
    with tempfile.TemporaryDirectory(prefix='bandit-native-check-') as directory:
        root = Path(directory)
        (root / 'unsafe.py').write_text('user_value = input()\neval(user_value)\n', encoding='utf-8')
        unsafe = run_bandit(root, ['unsafe.py'], timeout_seconds=30,
                            head=head, archive_sha256=archive_hash)
        if unsafe.status != 'success' or unsafe.version != version:
            raise RuntimeError(f'known unsafe fixture did not produce success: {unsafe.status}')
        if not any(f.test_id == 'B307' and f.file == 'unsafe.py' for f in unsafe.findings):
            raise RuntimeError('known eval fixture did not produce B307 at repo-relative unsafe.py')
        payload = unsafe.payload(2000)
        if not any(item['test_id'] == 'B307' and item['file'] == 'unsafe.py' for item in payload):
            raise RuntimeError('known issue was not present in the bounded model signal payload')

        (root / 'clean.py').write_text('def add(left, right):\n    return left + right\n', encoding='utf-8')
        clean = run_bandit(root, ['clean.py'], timeout_seconds=30,
                           head=head, archive_sha256=archive_hash)
        if clean.status != 'no_findings' or clean.findings:
            raise RuntimeError(f'clean fixture did not produce no_findings: {clean.status}')
    return {'unsafe_status': unsafe.status, 'unsafe_findings': [f.model_dump() for f in unsafe.findings],
            'unsafe_payload': payload, 'clean_status': clean.status}


def check_saved_pr(run_path, snapshot_dir, version):
    run = BaselineRun.model_validate_json(Path(run_path).read_bytes())
    artifact_root = Path(snapshot_dir)
    inp = ReviewInput.model_validate(json.loads((artifact_root / 'input.json').read_bytes()))
    if inp.pr.pr_number != 378 or inp.pr.head != run.result.pr.head:
        raise RuntimeError('saved input/run do not identify the requested frozen PR 378 head')
    archive_path = artifact_root / 'head.tar.gz'
    if sha256(archive_path.read_bytes()) != run.retrieval.archive_sha256:
        raise RuntimeError('cached frozen HEAD archive hash differs from saved review provenance')
    snapshot = Snapshot(inp.pr.repo, inp.pr.head, run.retrieval.archive_sha256,
                        archive_path, artifact_root / 'head')
    sources = snapshot_sources(snapshot, inp.pr.head)
    parsed = parse_diff(inp.diff)
    changed = {f.new_path for f in parsed if f.new_path and f.new_path.endswith('.py')}
    selected_context = {c.file for c in run.retrieval.chunks
                        if c.selected and c.file.endswith('.py')}
    expected = sorted(path for path in (changed | selected_context) & sources.keys() if eligible(path))
    result = analyze_snapshot(inp, snapshot, run.retrieval,
                              timeout_seconds=60,
                              budget=load_config('configs/experiment.toml').static_analysis.signal_token_budget)
    if result.version != version or result.status not in ('success', 'no_findings'):
        raise RuntimeError(f'PR static analysis failed: status={result.status}, error={result.error}')
    if result.analyzed_files != expected:
        raise RuntimeError('Bandit analyzed-file inventory differs from frozen changed/context selection')
    sent1, sent2 = result.payload(2000), result.payload(2000)
    if sent1 != sent2:
        raise RuntimeError('static signal payload is not deterministic')
    result = result.model_copy(update={'signals_sent': sent1})
    return result.model_dump()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='saved tools.json provenance from frozen PR 378')
    parser.add_argument('--snapshot-dir', required=True, help='existing cached acquisition artifact directory')
    args = parser.parse_args()
    version = get_bandit_version()
    if not version:
        parser.error('Bandit is unavailable; install the intended local version first')
    result = {'bandit_version': version,
              'known_fixtures': check_fixture(version),
              'pr_378_static_analysis': check_saved_pr(args.run, args.snapshot_dir, version)}
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == '__main__':
    main()
