"""No-model structural/provenance checks for prepared reviewer requests.

Golden text is deliberately not an input: source code may legitimately overlap
benchmark annotations. Trust comes from independent production preparation.
"""
import json

from reviewer.archives import sha256
from reviewer.schemas import FindingBatch


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _records(items, fields, name):
    if type(items) is not list:
        raise ValueError(f'invalid {name} evidence')
    for item in items:
        if type(item) is not dict or set(item) != set(fields):
            raise ValueError(f'unexpected {name} evidence fields')
        if any(type(item[key]) is not kind for key, kind in fields.items()):
            raise ValueError(f'invalid nested {name} evidence')


def validate_reviewer_payloads(payloads, *, frozen_diff, diff_sha256, context,
                              signals, prompt, model, config):
    """Validate exact A/B/C/D requests against independently prepared evidence.

    Strings remain opaque source text, including URLs or JSON-looking code.
    Context/signal records have closed shapes; evaluation objects cannot be
    smuggled in as additional keys or nested values.
    """
    if sha256(frozen_diff.encode('utf-8')) != diff_sha256:
        raise ValueError('frozen diff provenance mismatch')
    _records(context, {'file': str, 'start_line': int, 'end_line': int, 'content': str}, 'context')
    _records(signals, {'test_id': str, 'severity': str, 'confidence': str,
                      'file': str, 'line': int, 'message': str}, 'static signal')
    if type(payloads) is not dict or set(payloads) != {'baseline', 'retrieval', 'tools', 'full'}:
        raise ValueError('unexpected reviewer modes')
    for mode, payload in payloads.items():
        evidence = {'diff': frozen_diff}
        if mode != 'baseline':
            evidence['repository_context'] = context
        if mode in ('tools', 'full'):
            evidence['static_analysis_signals'] = signals
        expected = {
            'model': model, 'instructions': prompt,
            'input': [{'role': 'user', 'content': json.dumps(evidence, ensure_ascii=False)}],
            'text': {'format': {'type': 'json_schema', 'name': 'review_findings',
                               'strict': True, 'schema': FindingBatch.model_json_schema()}},
            'max_output_tokens': config.max_output_tokens,
            'store': False, 'truncation': 'disabled',
        }
        if config.temperature is not None:
            expected['temperature'] = config.temperature
        # Decode evidence for structural equality, without allowing additional
        # envelope fields, messages, roles, schema metadata or instructions.
        try:
            actual = json.loads(payload['input'][0]['content'])
            normalized = json.loads(_json(payload))
            normalized['input'][0]['content'] = actual
            expected['input'][0]['content'] = evidence
        except (KeyError, IndexError, TypeError, ValueError):
            raise ValueError(f'invalid {mode} reviewer payload') from None
        if _json(normalized) != _json(expected):
            raise ValueError(f'{mode} reviewer payload provenance mismatch')
