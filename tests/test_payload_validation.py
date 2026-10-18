"""Offline evidence boundary regressions; no model transport is constructed."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from reviewer.archives import sha256
from reviewer.config import load_config
from reviewer.models import build_review_payload
from reviewer.payload_validation import validate_reviewer_payloads
from reviewer.pipeline import review_prompt
from reviewer.schemas import ReviewInput
from reviewer.verification import _evidence
from test_github import DIFF
from test_schemas import pr_data
from test_verification import proposal


class PayloadValidationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(Path(__file__).resolve().parents[1] / 'configs/experiment.toml').review
        self.diff = DIFF.decode()
        self.context = [{'file': 'src/helper.py', 'start_line': 1, 'end_line': 1,
                         'content': 'https://example.test/documentation'}]
        self.signals = [{'test_id': 'B307', 'severity': 'MEDIUM', 'confidence': 'HIGH',
                         'file': 'src/a.py', 'line': 1, 'message': 'Use of eval detected.'}]
        self.guard = patch('reviewer.models.OpenAIReviewer.__init__',
                           side_effect=AssertionError('model client forbidden'))
        self.guard.start(); self.addCleanup(self.guard.stop)

    def payloads(self):
        return {mode: build_review_payload('gpt-6-luna', self.config, self.diff, review_prompt(),
                    context=deepcopy(self.context) if mode != 'baseline' else None,
                    static_signals=deepcopy(self.signals) if mode in ('tools', 'full') else None)
                for mode in ('baseline', 'retrieval', 'tools', 'full')}

    def validate(self, payloads, **updates):
        kwargs = dict(frozen_diff=self.diff, diff_sha256=sha256(self.diff.encode()),
                      context=self.context, signals=self.signals, prompt=review_prompt(),
                      model='gpt-6-luna', config=self.config)
        validate_reviewer_payloads(payloads, **(kwargs | updates))

    def test_golden_changes_do_not_change_reviewer_or_verifier_evidence(self):
        golden = {'findings': [{'message': self.diff, 'tp_fp': 'tp'},
                               {'message': self.context[0]['content'], 'tp_fp': 'fp'}]}
        before = self.payloads()
        inp = ReviewInput(pr=pr_data(), diff=self.diff, diff_complete=True)
        verifier_before = _evidence(inp, proposal(), None, None, 2000)
        golden['findings'][0].update(message='SECRET_GOLDEN_SENTINEL', tp_fp='fp',
                                    category='security', scope='introduced-by-pr')
        self.assertEqual(before, self.payloads())
        self.assertEqual(verifier_before, _evidence(inp, proposal(), None, None, 2000))
        self.assertNotIn('SECRET_GOLDEN_SENTINEL', json.dumps(before))
        self.validate(before)
        contexts = [json.loads(before[m]['input'][0]['content'])['repository_context']
                    for m in ('retrieval', 'tools', 'full')]
        self.assertEqual(contexts, [self.context] * 3)

    def test_nested_evaluation_metadata_and_evidence_mutations_rejected(self):
        for target in ('diff', 'context', 'signals', 'nested_context', 'nested_signals', 'golden'):
            with self.subTest(target=target):
                payloads = self.payloads()
                evidence = json.loads(payloads['tools']['input'][0]['content'])
                if target == 'diff': evidence['diff'] += 'injected'
                elif target == 'context': evidence['repository_context'][0]['content'] = 'injected'
                elif target == 'signals': evidence['static_analysis_signals'][0]['message'] = 'injected'
                elif target == 'nested_context': evidence['repository_context'][0]['judge_output'] = {'tp': True}
                elif target == 'nested_signals': evidence['static_analysis_signals'][0]['scope'] = 'introduced-by-pr'
                else: evidence['golden'] = {'findings': []}
                payloads['tools']['input'][0]['content'] = json.dumps(evidence)
                with self.assertRaisesRegex(ValueError, 'provenance mismatch'):
                    self.validate(payloads)

    def test_preparation_outputs_also_have_closed_shapes(self):
        for field in ('context', 'signals'):
            for mutation in ('extra_key', 'nested_value'):
                with self.subTest(field=field, mutation=mutation):
                    values = deepcopy(getattr(self, field))
                    if mutation == 'extra_key': values[0]['evaluation'] = {'tp_fp': 'tp'}
                    else: values[0]['file'] = {'golden': 'src/a.py'}
                    with self.assertRaises(ValueError):
                        self.validate(self.payloads(), **{field: values})

    def test_prompt_schema_envelope_and_diff_hash_provenance(self):
        for target in ('prompt', 'schema', 'extra_message', 'metadata'):
            with self.subTest(target=target):
                payloads = self.payloads(); p = payloads['baseline']
                if target == 'prompt': p['instructions'] += 'injected'
                elif target == 'schema': p['text']['format']['schema']['description'] = 'golden'
                elif target == 'extra_message': p['input'].append({'role': 'user', 'content': 'golden'})
                else: p['evaluation'] = {'score': 1}
                with self.assertRaises(ValueError): self.validate(payloads)
        with self.assertRaisesRegex(ValueError, 'diff provenance'):
            self.validate(self.payloads(), diff_sha256='0' * 64)
