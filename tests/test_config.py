from pathlib import Path
from tempfile import TemporaryDirectory
import tomllib
import unittest

from pydantic import ValidationError

from reviewer.config import ExperimentConfig, load_config, resolve_models, validate_ready

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "experiment.toml"


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG)

    def test_template_loads_without_models(self):
        self.assertEqual(self.config.repetitions, 3)
        self.assertFalse(self.config.evaluation.human_audit_required)
        self.assertIsNone(resolve_models(self.config, environment={}).review)

    def test_bad_settings(self):
        cases = [("repetitions", 0), ("seed", True), ("schema_version", True),
                 ("review.max_attempts", -1), ("review.max_output_tokens", "4096"),
                 ("review.temperature", float("inf")), ("review.request_byte_budget", -1),
                 ("retrieval.max_chunks", 0), ("static_analysis.timeout_seconds", 0),
                 ("verification.uncertain_policy", "guess"),
                 ("verification.uncertain_policy", "retain"),
                 ("evaluation.development_repository_fraction", 1.0),
                 ("evaluation.human_audit_required", 0),
                 ("benchmark.categories", []), ("benchmark.scopes", ["introduced-by-pr"] * 2),
                 ("benchmark.revision", "main"), ("benchmark.manifest_sha256", "bad"),
                 ("models.review_model_env", "BAD NAME"), ("unexpected", "secret")]
        for dotted, value in cases:
            data = self.config.model_dump()
            parts = dotted.split("."); target = data
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = value
            with self.subTest(field=dotted), self.assertRaises(ValidationError):
                ExperimentConfig.model_validate(data)

    def test_all_mode_flags_are_fixed(self):
        for mode in self.config.modes:
            for flag in ("retrieval", "static_analysis", "verification"):
                data = self.config.model_dump()
                data["modes"][mode][flag] = not data["modes"][mode][flag]
                with self.subTest(mode=mode, flag=flag), self.assertRaises(ValidationError):
                    ExperimentConfig.model_validate(data)
        data = self.config.model_dump(); del data["modes"]["full"]
        with self.assertRaises(ValidationError):
            ExperimentConfig.model_validate(data)
        data = self.config.model_dump(); data["modes"]["baseline"]["retrieval"] = 0
        with self.assertRaises(ValidationError):
            ExperimentConfig.model_validate(data)

    def test_environment_and_explicit_override_without_secrets(self):
        env = {"REVIEW_MODEL": "env-model", "REVIEW_PROVIDER": "example",
               "API_KEY": "must-not-be-recorded", "VERIFY_MODEL": "", "VERIFY_PROVIDER": ""}
        resolved = resolve_models(self.config, environment=env, overrides={"review_model": "override"})
        self.assertEqual(resolved.review.model, "override")
        self.assertEqual(resolved.review.provider, "example")
        self.assertNotIn("must-not-be-recorded", resolved.model_dump_json())
        with self.assertRaises(ValueError):
            resolve_models(self.config, environment={}, overrides={"api_key": "secret"})
        with self.assertRaises(ValueError):
            resolve_models(self.config, environment={"REVIEW_MODEL": "missing-provider"})
        self.config.models.review_model_env = "CUSTOM_MODEL"
        self.assertEqual(resolve_models(self.config, environment={"CUSTOM_MODEL": "custom",
                         "REVIEW_PROVIDER": "example"}).review.model, "custom")

    def test_mode_specific_readiness(self):
        models = resolve_models(self.config, environment={"REVIEW_MODEL": "model", "REVIEW_PROVIDER": "example"})
        self.config.review.request_byte_budget = 0
        with self.assertRaisesRegex(ValueError, "serialized-request byte estimator"):
            validate_ready(self.config, models, "baseline")
        self.config.review.request_byte_budget = 16000
        validate_ready(self.config, models, "baseline")
        validate_ready(self.config, models, "retrieval")  # Structural/lexical needs no embedding API.
        with self.assertRaisesRegex(ValueError, "verify"):
            validate_ready(self.config, models, "full")
        with self.assertRaisesRegex(ValueError, "unknown"):
            validate_ready(self.config, models, "invalid")
        with self.assertRaisesRegex(ValueError, "split manifest"):
            validate_ready(self.config, models, "baseline", final_experiment=True)
        self.config.benchmark.split_manifest = "frozen.json"
        with self.assertRaisesRegex(ValueError, "deferred"):
            validate_ready(self.config, models, "baseline", final_experiment=True)

    def test_full_readiness_with_explicit_settings(self):
        self.config.review.request_byte_budget = 16000
        self.config.verification.uncertain_policy = "drop"
        env = {f"{role}_{field}": "example" for role in ("REVIEW", "VERIFY", "EMBEDDING")
               for field in ("MODEL", "PROVIDER")}
        validate_ready(self.config, resolve_models(self.config, environment=env), "full")

    def test_optional_context_cannot_fill_entire_request_byte_budget(self):
        self.config.review.request_byte_budget = 6000
        env = {f"{role}_{field}": "example" for role in ("REVIEW", "EMBEDDING")
               for field in ("MODEL", "PROVIDER")}
        with self.assertRaisesRegex(ValueError, "leave room"):
            validate_ready(self.config, resolve_models(self.config, environment=env), "retrieval")

    def test_toml_errors_and_missing_files(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            with self.assertRaises(FileNotFoundError):
                load_config(path)
            path.write_text("[broken")
            with self.assertRaises(tomllib.TOMLDecodeError):
                load_config(path)
            path.write_text("seed = 42")
            with self.assertRaises(ValidationError):
                load_config(path)
