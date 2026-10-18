"""Deterministic application-level request size estimator (not tokenizer usage)."""
import json

SAFETY_MARGIN_BYTES = 4096
BUDGET_METHOD = 'serialized_request_body_utf8_bytes_plus_4096_safety_margin'
BUDGET_UNIT = 'serialized_request_estimator_units'


def measure_request(payload: dict, ceiling: int | None = None) -> dict:
    """Measure the exact compact JSON encoding used by the HTTP transport."""
    body = json.dumps(payload, allow_nan=False).encode('utf-8')
    estimate = len(body) + SAFETY_MARGIN_BYTES
    result = {
        'serialized_request_body_utf8_bytes': len(body),
        'safety_margin_bytes': SAFETY_MARGIN_BYTES,
        'estimator_value': estimate,
        'estimator_units': BUDGET_UNIT,
        'method': BUDGET_METHOD,
    }
    if ceiling is not None:
        result['configured_estimator_ceiling'] = ceiling
        result['fits'] = estimate <= ceiling
    return result
