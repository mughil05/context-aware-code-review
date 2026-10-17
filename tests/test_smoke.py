"""Offline CLI entry-point checks; no live service calls."""

import subprocess
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ScaffoldSmokeTests(unittest.TestCase):
    def invoke(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "reviewer", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_help_is_available_without_credentials(self):
        result = self.invoke("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("baseline", result.stdout)

    def test_review_requires_explicit_ready_configuration(self):
        result = self.invoke("review", "--repo", "https://github.com/example/project", "--pr", "123")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Configuration error", result.stderr)

    def test_evaluation_cannot_silently_report_success(self):
        result = self.invoke("evaluate", "--benchmark", "reviewbench")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--benchmark-archive", result.stderr)

    def test_invalid_mode_is_rejected(self):
        result = self.invoke("evaluate", "--benchmark", "reviewbench", "--modes", "invalid")
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid choice", result.stderr)

    def test_config_has_the_four_ablation_modes(self):
        with (ROOT / "configs" / "experiment.toml").open("rb") as stream:
            config = tomllib.load(stream)
        flags = ("retrieval", "static_analysis", "verification")
        observed = {name: tuple(mode[key] for key in flags) for name, mode in config["modes"].items()}
        self.assertEqual(observed, {
            "baseline": (False, False, False),
            "retrieval": (True, False, False),
            "tools": (True, True, False),
            "full": (True, True, True),
        })


if __name__ == "__main__":
    unittest.main()
