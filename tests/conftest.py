import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(ROOT, "tools", "fixtures")

sys.path.insert(0, ROOT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Numeric fields of tools/fixtures/metadata.json, in file order
FIXTURE_NUMERIC_KEYS = [
    "faceCount", "girdles", "lengthWidthRatio", "tableWidthRatio",
    "culetWidthRatio", "pavilionWidthRatio", "crownWidthRatio",
    "volumeWidthCubedRatio",
]


@pytest.fixture
def run_script(tmp_path):
    """Run a repository script (or `-c` code) from an empty working directory."""

    def run(args, stdin="", cwd=None):
        env = dict(os.environ, PYTHONPATH=ROOT, MPLBACKEND="Agg")
        if args[0].endswith(".py"):
            args = [os.path.join(ROOT, args[0])] + list(args[1:])
        result = subprocess.run(
            [sys.executable] + list(args), input=stdin, text=True,
            capture_output=True, cwd=cwd or tmp_path, env=env, timeout=600,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result

    return run
