"""Lab 11 - production hardening: tests, CI, a containerized service."""

import os
import re
import subprocess

import pytest
import requests
import yaml

from fraudlab.config import REPO_ROOT

TESTS = REPO_ROOT / "tests"


def _pytest(*args):
    return subprocess.run(["uv", "run", "--quiet", "pytest", "tests", "-q", "-p", "no:cacheprovider", *args],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=900)


def test_you_wrote_tests():
    files = sorted(TESTS.glob("test_*.py")) if TESTS.exists() else []
    assert len(files) >= 2, "write at least two test files in tests/ (e.g. test_features.py, test_contracts.py)"
    n = sum(len(re.findall(r"^\s*def test_", f.read_text(), re.M)) for f in files)
    assert n >= 6, f"found {n} test functions; aim for at least 6"


def test_unit_tests_pass():
    r = _pytest("-m", "not services")
    assert r.returncode == 0, f"`pytest tests -m 'not services'` failed:\n{r.stdout[-2500:]}"


def test_contract_tests_pass():
    r = _pytest("-m", "services")
    assert r.returncode != 5, "no tests marked @pytest.mark.services - add contract tests against the platform"
    assert r.returncode == 0, f"`pytest tests -m services` failed:\n{r.stdout[-2500:]}"


def test_ci_workflow():
    files = list((REPO_ROOT / ".github" / "workflows").glob("*.y*ml"))
    assert files, "add a GitHub Actions workflow in .github/workflows/"
    text = "\n".join(f.read_text() for f in files)
    for f in files:
        yaml.safe_load(f.read_text())
    assert "pytest" in text and "ruff" in text, "CI should lint (ruff) and run the unit tests (pytest)"
    assert "not services" in text, "CI has no platform: run only tests that don't need services"


def test_serving_container():
    assert (REPO_ROOT / "docker" / "serving" / "Dockerfile").exists(), "add docker/serving/Dockerfile"
    cfg = subprocess.run(["docker", "compose", "--profile", "serving", "config", "--format", "json"],
                         cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
    assert '"serving"' in cfg.stdout, "add a `serving` service (profile: serving) - see compose.override.yaml"
    port = os.environ.get("SERVING_PORT", "8001")
    try:
        body = requests.get(f"http://localhost:{port}/health", timeout=5).json()
    except requests.RequestException:
        pytest.fail(f"containerized API not answering on :{port}. `docker compose --profile serving up -d --build serving`")
    assert body["status"] == "ok"
