import os

import pytest
from packaging.requirements import Requirement

from conftest import ROOT

LINUX = {"sys_platform": "linux", "platform_system": "Linux"}
MACOS = {"sys_platform": "darwin", "platform_system": "Darwin"}


def requirements():
    with open(os.path.join(ROOT, "requirements.txt")) as f:
        lines = [ln.split("#")[0].strip() for ln in f]
    return {r.name.lower(): r for r in map(Requirement, filter(None, lines))}


def applies(req, env):
    return req.marker is None or req.marker.evaluate(env)


@pytest.mark.parametrize("name", ["tensorflow-macos", "tensorflow-metal"])
def test_apple_packages_are_macos_only(name):
    """#6: no Linux wheels exist for these; they must not install there."""
    req = requirements()[name]
    assert not applies(req, LINUX)
    assert applies(req, MACOS)


def test_tensorflow_versions_match():
    reqs = requirements()
    assert str(reqs["tensorflow"].specifier) == "==2.14.0"
    assert str(reqs["tensorflow-macos"].specifier) == "==2.14.0"


def test_numpy_is_pinned_below_2():
    """#6: TensorFlow 2.14 fails to import under numpy 2."""
    spec = requirements()["numpy"].specifier
    assert "1.26.4" in spec
    assert "2.0.0" not in spec


def test_no_unrelated_image_package():
    """#9: PyPI "Image" is a Django app; PIL.Image comes from Pillow."""
    reqs = requirements()
    assert "image" not in reqs
    assert "pillow" in reqs


def test_installed_environment_imports():
    """The environment installed from requirements.txt imports cleanly."""
    import numpy
    import tensorflow

    assert int(numpy.__version__.split(".")[0]) < 2
    assert tensorflow.__version__ == "2.14.0"
