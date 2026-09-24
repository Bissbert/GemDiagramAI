import json
import os

import numpy as np
import pytest
from PIL import Image

from conftest import FIXTURES, FIXTURE_NUMERIC_KEYS

# prepare_data_for_training.py at a 16 px image size, so the models stay tiny
PREPARE_TINY = (
    "import data_utils; data_utils.IMAGE_SIZE = 16; "
    "import prepare_data_for_training as p; p.main()"
)


def fixture_answers():
    return f"{os.path.join(FIXTURES, 'svg')}\n{os.path.join(FIXTURES, 'metadata.json')}\n"


def write_stats(directory, keys, mean, std):
    directory.mkdir(parents=True, exist_ok=True)
    stats = {"keys": keys, "mean": dict(zip(keys, mean)), "std": dict(zip(keys, std))}
    (directory / "metadata_stats.json").write_text(json.dumps(stats))


def test_prepare_training_on_fixtures(run_script, tmp_path):
    """Full-size preparation writes a stats file listing the numeric fields."""
    out = tmp_path / "training"
    run_script(["prepare_data_for_training.py", "--save_dir", str(out)],
               stdin=fixture_answers())

    imgs = np.load(out / "training_data_imgs.npz")["arr_0"]
    assert imgs.shape == (3, 512, 512, 3)
    stats = json.loads((out / "metadata_stats.json").read_text())
    assert stats["keys"] == FIXTURE_NUMERIC_KEYS
    for key in FIXTURE_NUMERIC_KEYS:
        column = np.load(out / f"training_data_meta_{key}.npz")["arr_0"]
        assert column.dtype == np.float64 and column.shape == (3,)


def test_generation_prep_writes_into_save_dir(run_script, tmp_path):
    """#5: --save_dir is honoured and nothing lands in the working directory."""
    training = tmp_path / "training"
    write_stats(training, ["a", "b", "c"], [10.0, 0.5, 3.0], [2.0, 0.25, 0.0])
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    save_dir = tmp_path / "does" / "not" / "exist"

    run_script(["prepare_data_for_generation.py", "--save_dir", str(save_dir),
                "--training_data_dir", str(training)],
               stdin="14\n\n9\n", cwd=cwd)

    saved = np.load(save_dir / "generation_metadata.npz")
    # a: (14-10)/2; b: blank -> mean -> 0; c: std 0 -> 0
    np.testing.assert_allclose(saved["arr_0"], [[2.0, 0.0, 0.0]])
    assert saved["arr_0"].dtype == np.float32
    assert list(saved["keys"]) == ["a", "b", "c"]
    assert not list(cwd.glob("*.npz"))


def test_generation_prep_without_stats_explains(tmp_path):
    import subprocess
    import sys

    from conftest import ROOT

    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, "prepare_data_for_generation.py"),
         "--training_data_dir", str(tmp_path / "missing")],
        input="", text=True, capture_output=True, cwd=tmp_path)
    assert result.returncode != 0
    assert "prepare_data_for_training.py" in result.stderr


def test_end_to_end_conditioned_generation(run_script, tmp_path):
    """#4 and #5 together: prepare -> train -> prepare generation -> run."""
    training = tmp_path / "training"
    run_script(["-c", PREPARE_TINY, "--save_dir", str(training)],
               stdin=fixture_answers())
    assert np.load(training / "training_data_imgs.npz")["arr_0"].shape == (3, 16, 16, 3)

    work = tmp_path / "work"
    work.mkdir()
    run_script(["train_model.py", "--training_data_dir", str(training),
                "--epochs", "1", "--batch_size", "2", "--save_interval", "1"],
               cwd=work)
    model_path = work / "generator_model_final.h5"
    assert model_path.exists()

    generation = tmp_path / "generation"
    answers = "".join(f"{v}\n" for v in (65, 16, 1.0, "", "", 0.5, 0.17, 0.2))
    run_script(["prepare_data_for_generation.py", "--save_dir", str(generation),
                "--training_data_dir", str(training)], stdin=answers, cwd=work)

    output = tmp_path / "diagram.png"
    run_script(["run_model.py", "--model", str(model_path),
                "--generation_data_dir", str(generation), "--output", str(output)],
               cwd=work)
    with Image.open(output) as img:
        assert img.size == (16, 16)

    from tensorflow.keras.models import load_model

    generator = load_model(model_path, compile=False)
    assert generator.inputs[1].shape[-1] == len(FIXTURE_NUMERIC_KEYS)


def test_run_model_rejects_wrong_metadata_width(run_script, tmp_path):
    from model import build_generator

    model_path = tmp_path / "g.h5"
    build_generator(8, 3, image_size=16).save(model_path)
    generation = tmp_path / "generation"
    generation.mkdir()
    np.savez_compressed(generation / "generation_metadata.npz",
                        np.zeros((1, 5), np.float32))

    with pytest.raises(AssertionError, match="the model expects 3"):
        run_script(["run_model.py", "--model", str(model_path),
                    "--generation_data_dir", str(generation),
                    "--output", str(tmp_path / "x.png")])
