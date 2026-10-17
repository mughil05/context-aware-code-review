"""Run our exports through actual pinned upstream validation; no judge imports."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest

from reviewer.benchmark import Subset, export_predictions
from reviewer.schemas import BaselineRun, Finding, PRMetadata, ReviewResult

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/reviewbench-validation'
BRIDGE = ROOT / 'tests/reviewbench_validate.mjs'
SUBSET = ROOT / 'configs/reviewbench-dev5.json'
LABEL = 'GROUND_TRUTH_SENTINEL_482601'


def synthetic_run(task, *, empty=False):
    # Only the existing result schema and adapter are used. No provider object,
    # model transport, repository fetching, or baseline generation is invoked.
    f = Finding(title='Synthetic issue', category='correctness', severity='medium',
                file='src/dir with space/δ.py', start_line=17, end_line=19,
                description='Synthetic description with newline\nand Unicode: café.',
                evidence=['Evidence <&> "quoted"', 'Second fixture evidence'],
                suggested_fix='A synthetic suggested fix.', confidence=0.8)
    result = ReviewResult(pr=task.pr, mode='baseline', status='success',
                          findings=[] if empty else [f])
    return BaselineRun(result=result, diff_base=task.pr.base, diff_sha256='a'*64,
                       requested_model='offline-synthetic', settings={'synthetic': True},
                       prompt_sha256='b'*64, prompt_text='Synthetic test only', output_schema={},
                       request_sha256=None, created_at='2026-10-15T00:00:00+00:00', attempts=[],
                       model_latency_seconds=0.0, acquisition_latency_seconds=0.0,
                       total_latency_seconds=0.0, output_text=LABEL,
                       input_artifact_directory=LABEL)


class ReviewBenchCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.node = shutil.which('node')
        if not cls.node:
            raise unittest.SkipTest('Node 24+ required for upstream validator compatibility tests')
        version = subprocess.run([cls.node, '--version'], capture_output=True, text=True,
                                 check=True, env={'PATH': os.environ.get('PATH', '')}).stdout
        if int(version.strip().lstrip('v').split('.')[0]) < 24:
            raise unittest.SkipTest('Node 24+ required for native TypeScript validation fixtures')
        cls.tasks = Subset.model_validate_json(SUBSET.read_bytes()).tasks

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.candidates = self.root / 'candidates'
        self.candidates.mkdir()

    def populate(self, *, empty=False):
        exports = {}
        for i, task in enumerate(self.tasks):
            exported = export_predictions(synthetic_run(task, empty=empty))
            # Deliberately unrelated names and nested folders verify content-based ID.
            parent = self.candidates / f'nested-{i}'
            parent.mkdir()
            path = parent / f'arbitrary-{i}.json'
            path.write_text(json.dumps(exported, ensure_ascii=False), encoding='utf-8')
            exports[task.task_id] = (path, exported)
        return exports

    def validate(self, path=None, *, allow_empty=False):
        command = [self.node, '--permission', f'--allow-fs-read={ROOT}',
                   f'--allow-fs-read={self.root}', str(BRIDGE), str(path or self.candidates), str(SUBSET)]
        if allow_empty:
            command.append('--allow-empty')
        # No credentials or NODE_OPTIONS inherited. No child-process/write
        # permissions; runtime imports are confined to the pinned validator.
        output = subprocess.run(command, capture_output=True, text=True, timeout=20,
                                env={'PATH': os.environ.get('PATH', '')})
        self.assertIn(output.returncode, (0, 1), output.stderr)
        self.assertTrue(output.stdout, output.stderr)
        return output.returncode, json.loads(output.stdout)

    def test_vendored_validation_files_are_exact_pinned_sources(self):
        known = {
            'scripts/eval/input-validation.ts': '30bc63f53695d83c468aba4d22c3e49da5dc1b68dc98f0e88f394aea313fb153',
            'scripts/lib/types.ts': 'f835f80db8c4441217efc2e0b2d0214632bac1866af98679c64eeb67860b30d0',
        }
        metadata = json.loads((FIXTURE / 'provenance.json').read_text())
        self.assertEqual(metadata['revision'], '968291d99e82b3e2969d7e0b4acca62585886c13')
        for path, sha in known.items():
            self.assertEqual(hashlib.sha256((FIXTURE / path).read_bytes()).hexdigest(), sha)
            self.assertEqual(metadata['sha256'][path], sha)
        self.assertEqual(sorted(p.name for p in FIXTURE.rglob('*.ts')), ['input-validation.ts', 'types.ts'])

    def test_actual_loader_preserves_all_five_ids_coordinates_and_messages(self):
        exports = self.populate()
        code, result = self.validate()
        self.assertEqual(code, 0, result)
        self.assertEqual(result['problems'], [])
        self.assertEqual(result['summary']['prs_to_evaluate'], 5)
        self.assertEqual(result['stats']['files_loaded'], 5)
        self.assertEqual(set(result['loaded']), set(exports))
        for key, (_, exported) in exports.items():
            loaded = result['loaded'][key]
            self.assertEqual(loaded['pr'], exported['pr'])
            self.assertEqual(loaded['findings'], exported['findings'])
            self.assertEqual(loaded['usage'], exported['usage'])
            self.assertEqual(loaded['findings'][0]['message'],
                'Synthetic issue\n\nSynthetic description with newline\nand Unicode: café.'
                '\n\nEvidence:\n- Evidence <&> "quoted"\n- Second fixture evidence'
                '\n\nSuggested fix:\nA synthetic suggested fix.')
        self.assertNotIn(LABEL, json.dumps(result))

    def test_export_allowlist_excludes_labels_raw_response_and_provenance(self):
        run = synthetic_run(self.tasks[0])
        # A separately loaded reference must never be serialized with predictions.
        reference = {'findings': [{'message': LABEL, 'tp_fp': 'tp'}]}
        exported = export_predictions(run)
        self.assertEqual(set(exported), {'pr', 'agent', 'findings', 'usage'})
        self.assertEqual(set(exported['findings'][0]),
                         {'producer', 'file', 'start_line', 'end_line', 'message'})
        self.assertNotIn(reference['findings'][0]['message'], json.dumps(exported))
        for label in ('tp_fp', 'golden', 'scope', 'context_required', 'matched_golden_index'):
            self.assertNotIn(label, json.dumps(exported))

    def test_all_zero_reviews_require_allow_empty_but_keep_five_tasks(self):
        self.populate(empty=True)
        code, result = self.validate()
        self.assertEqual(code, 1)
        self.assertIn('candidate input produced 0 findings', result['problems'])
        code, result = self.validate(allow_empty=True)
        self.assertEqual(code, 0, result)
        self.assertEqual(result['summary']['prs_to_evaluate'], 5)
        self.assertEqual(result['summary']['candidate_findings'], 0)

    def test_mixed_zero_review_is_not_dropped(self):
        exports = self.populate()
        path, data = exports[self.tasks[1].task_id]
        data['findings'] = []
        path.write_text(json.dumps(data))
        code, result = self.validate()
        self.assertEqual(code, 0, result)
        self.assertEqual(result['summary']['prs_to_evaluate'], 5)
        self.assertEqual(result['loaded'][self.tasks[1].task_id]['findings'], [])

    def test_wrong_base_is_rejected_even_when_task_key_matches(self):
        exports = self.populate()
        path, data = exports[self.tasks[0].task_id]
        data['pr']['base'] = '0'*40
        path.write_text(json.dumps(data))
        code, result = self.validate()
        self.assertEqual(code, 1)
        self.assertTrue(any('differs from golden on: base' in p for p in result['problems']))

    def test_malformed_input_and_missing_golden_are_rejected(self):
        exports = self.populate()
        path, data = exports[self.tasks[0].task_id]
        data['pr']['head'] = '0'*40
        path.write_text(json.dumps(data))
        code, result = self.validate()
        self.assertEqual(code, 1)
        self.assertEqual(result['summary']['missing_golden'], 1)
        path.write_text('{')
        code, result = self.validate()
        self.assertEqual(code, 1)
        self.assertEqual(result['stats']['skipped_malformed_files'], 1)
        data['findings'][0]['start_line'] = '17'
        path.write_text(json.dumps(data))
        code, result = self.validate()
        self.assertEqual(code, 1)
        self.assertEqual(result['stats']['skipped_invalid_files'], 1)

    def test_duplicate_pr_files_are_concatenated_by_upstream(self):
        exports = self.populate()
        data = exports[self.tasks[0].task_id][1]
        (self.candidates / 'duplicate.json').write_text(json.dumps(data))
        code, result = self.validate()
        self.assertEqual(code, 0, result)
        self.assertEqual(result['stats']['files_loaded'], 6)
        self.assertEqual(len(result['loaded'][self.tasks[0].task_id]['findings']), 2)

    def test_whole_run_directory_is_not_valid_candidate_input(self):
        self.populate()
        (self.root / 'summary.json').write_text('{"scoring_status":"paused"}')
        code, result = self.validate(self.root)
        self.assertEqual(code, 1)
        self.assertEqual(result['stats']['skipped_invalid_files'], 1)


if __name__ == '__main__':
    unittest.main()
