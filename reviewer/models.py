"""One isolated OpenAI Responses adapter. No tools, context lookup, or API discovery."""

from dataclasses import dataclass
from http.client import HTTPException
import json
import os
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from pydantic import ValidationError

from reviewer.config import ReviewConfig
from reviewer.schemas import FindingBatch, ModelAttempt, TokenUsage


class ModelError(ValueError):
    """Sanitized failure independent of provider response bodies/request headers."""
    def __init__(self, code: str, message: str, *, retryable: bool = False):
        self.code = code
        self.retryable = retryable
        super().__init__(f"{code}: {message}")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _post(payload: dict, api_key: str, timeout: float) -> dict:
    request = Request("https://api.openai.com/v1/responses", method="POST",
                      data=json.dumps(payload, allow_nan=False).encode("utf-8"),
                      headers={"Authorization": "Bearer " + api_key,
                               "Content-Type": "application/json", "Accept-Encoding": "identity"})
    try:
        with build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
            if response.status != 200:
                raise ModelError("api_error", "unexpected response status")
            raw = response.read(8 * 1024 * 1024 + 1)
            if len(raw) > 8 * 1024 * 1024:
                raise ModelError("malformed_response", "API response exceeded byte limit")
            length = response.headers.get("Content-Length")
            if length is not None and (not length.isdecimal() or int(length) != len(raw)):
                raise ModelError("malformed_response", "incomplete API response")
            try:
                return json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                raise ModelError("malformed_response", "API response was not valid JSON") from None
    except HTTPError as error:
        with error:
            status = error.code
        if status in (401, 403):
            raise ModelError("authentication", "OpenAI denied access; check API key and permissions") from None
        if status == 429:
            raise ModelError("rate_limit", "OpenAI rate or quota limit reached", retryable=True) from None
        if status in (400, 404, 422):
            raise ModelError("unsupported_configuration", "OpenAI rejected the model, schema, input, or settings; check configuration") from None
        raise ModelError("api_error", f"OpenAI returned HTTP {status}", retryable=status >= 500) from None
    except TimeoutError:
        raise ModelError("timeout", "OpenAI request timed out; completion/billing is unknown") from None
    except URLError as error:
        if isinstance(error.reason, TimeoutError):
            raise ModelError("timeout", "OpenAI request timed out; completion/billing is unknown") from None
        raise ModelError("network", "OpenAI connection failed") from None
    except (OSError, HTTPException):
        raise ModelError("network", "OpenAI response could not be read completely") from None


@dataclass(frozen=True)
class ModelReply:
    text: str | None
    attempts: tuple[ModelAttempt, ...]
    latency_seconds: float
    error: str | None


class OpenAIReviewer:
    """Configuration is explicit; review payload contains only prompt/schema/diff.

    Temperature support is deliberately conservative: the documented non-reasoning
    GPT-4.1/GPT-4o families only. For other models, omit temperature entirely; the API
    must support Responses + strict structured text or return a clear failure.
    """
    def __init__(self, model: str, config: ReviewConfig, *, api_key: str | None = None,
                 transport=_post, clock=time.perf_counter, sleep=time.sleep):
        if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]+", model):
            raise ModelError("unsupported_configuration", "a valid explicit REVIEW_MODEL is required")
        self.config = ReviewConfig.model_validate(config.model_dump())
        if not 1 <= config.max_attempts <= 3:
            raise ModelError("unsupported_configuration", "max_attempts must be between 1 and 3")
        if config.temperature is not None and not re.fullmatch(
            r"(?:gpt-4\.1(?:-mini|-nano)?|gpt-4o(?:-mini)?)(?:-\d{4}-\d{2}-\d{2})?", model
        ):
            raise ModelError("unsupported_configuration", "omit temperature for models outside the supported sampling families")
        if config.input_token_limit <= 0:
            raise ModelError("unsupported_configuration", "set an explicit input token budget")
        key = os.environ.get("OPENAI_API_KEY", "") if api_key is None else api_key
        if not isinstance(key, str) or not key.strip() or any(ord(c) < 33 or ord(c) > 126 for c in key):
            raise ModelError("missing_api_key", "set OPENAI_API_KEY to a valid nonempty value")
        self.model = model
        self._api_key = key
        self._transport, self._clock, self._sleep = transport, clock, sleep

    def payload(self, diff: str, prompt: str, *, context: list[dict] | None = None) -> dict:
        # Explicit allowlist: never serialize ReviewInput/AcquiredPR or config wholesale.
        evidence = {"diff": diff}
        if context is not None:
            evidence['repository_context'] = context
        payload = {"model": self.model, "instructions": prompt,
                   "input": [{"role": "user", "content": json.dumps(evidence, ensure_ascii=False)}],
                   "text": {"format": {"type": "json_schema", "name": "review_findings",
                                       "strict": True, "schema": FindingBatch.model_json_schema()}},
                   "max_output_tokens": self.config.max_output_tokens,
                   "store": False, "truncation": "disabled"}
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        # Intentionally conservative size estimate, NOT measured token usage. Refuse
        # rather than truncate. API context checks remain authoritative for the model.
        byte_estimate = len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) + 4096
        if byte_estimate > self.config.input_token_limit:
            raise ModelError("input_budget", "request exceeds conservative input budget; diff was not truncated")
        return payload

    def complete(self, payload: dict) -> ModelReply:
        started = self._clock()
        attempts = []
        for number in range(1, self.config.max_attempts + 1):
            start = self._clock()
            response_id = returned_model = text = None
            usage = None
            failure = None
            response_status = incomplete_details = None
            try:
                response = self._transport(payload, self._api_key, self.config.timeout_seconds)
                if not isinstance(response, dict):
                    raise ModelError("malformed_response", "API response must be an object")
                status = response.get('status')
                response_status = status if isinstance(status, str) and status in (
                    'completed', 'failed', 'in_progress', 'cancelled', 'queued', 'incomplete'
                ) else ('unknown' if status is not None else None)
                details = response.get('incomplete_details')
                if details is not None:
                    reason = details.get('reason') if isinstance(details, dict) else None
                    incomplete_details = {'reason': reason if isinstance(reason, str) and reason in (
                        'max_output_tokens', 'content_filter') else 'unknown'}
                response_id, returned_model = response.get("id"), response.get("model")
                if (not isinstance(response_id, str) or not response_id.strip()
                        or not isinstance(returned_model, str) or not returned_model.strip()):
                    response_id = returned_model = None
                    raise ModelError("malformed_response", "API response omitted response/model identity")
                if response.get("usage") is not None:
                    raw = response["usage"]
                    try:
                        usage = TokenUsage(input_tokens=raw["input_tokens"], output_tokens=raw["output_tokens"],
                                           total_tokens=raw["total_tokens"],
                                           cached_input_tokens=raw.get("input_tokens_details", {}).get("cached_tokens"),
                                           reasoning_tokens=raw.get("output_tokens_details", {}).get("reasoning_tokens"))
                    except (KeyError, TypeError, AttributeError, ValidationError):
                        raise ModelError("malformed_response", "invalid token usage") from None
                if response.get("status") != "completed" or response.get("error") is not None:
                    raise ModelError("incomplete_response", "model response did not complete successfully")
                output = response.get("output")
                if not isinstance(output, list):
                    raise ModelError("malformed_response", "missing response output")
                parts = []
                for item in output:
                    if not isinstance(item, dict):
                        raise ModelError("malformed_response", "invalid output item")
                    if item.get("type") == "reasoning":
                        continue  # No reasoning content is persisted or treated as findings.
                    if item.get("type") != "message" or item.get("role") != "assistant" or item.get("status") != "completed":
                        raise ModelError("malformed_response", "unexpected output type or incomplete assistant message")
                    if not isinstance(item.get("content"), list):
                        raise ModelError("malformed_response", "invalid assistant message")
                    for part in item["content"]:
                        if not isinstance(part, dict):
                            raise ModelError("malformed_response", "invalid response content")
                        if part.get("type") == "refusal":
                            raise ModelError("refusal", "model refused the review")
                        if part.get("type") != "output_text" or not isinstance(part.get("text"), str):
                            raise ModelError("malformed_response", "expected structured text output")
                        parts.append(part["text"])
                if len(parts) != 1:
                    raise ModelError("malformed_response", "expected exactly one structured output")
                text = parts[0]
            except ModelError as error:
                failure = error
            attempts.append(ModelAttempt(number=number, latency_seconds=max(0.0, self._clock() - start),
                                         response_id=response_id, returned_model=returned_model,
                                         usage=usage, error_code=failure.code if failure else None,
                                         response_status=response_status, incomplete_details=incomplete_details))
            if failure is None:
                return ModelReply(text, tuple(attempts), max(0.0, self._clock() - started), None)
            if not failure.retryable or number == self.config.max_attempts:
                return ModelReply(None, tuple(attempts), max(0.0, self._clock() - started), str(failure))
            self._sleep(float(2 ** (number - 1)))
        raise AssertionError("unreachable")
