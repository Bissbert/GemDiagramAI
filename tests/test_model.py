import numpy as np
import pytest

import model
from model import build_combined, build_discriminator, build_generator

SIZE = 16
META = 3
Z = 8


def tiny_models(meta_dim=META):
    generator = build_generator(Z, meta_dim, image_size=SIZE)
    discriminator = build_discriminator((SIZE, SIZE, 3), meta_dim)
    combined = build_combined(generator, discriminator)
    return generator, discriminator, combined


def synthetic(n=4, meta_dim=META, seed=0):
    rng = np.random.default_rng(seed)
    imgs = rng.uniform(-1, 1, (n, SIZE, SIZE, 3)).astype("float32")
    metadata = rng.normal(0, 1, (n, meta_dim)).astype("float32")
    return imgs, metadata


def test_generator_output_depends_on_metadata():
    """#4: metadata is a generator input and changes the image."""
    generator = build_generator(Z, META, image_size=SIZE)
    assert [list(t.shape) for t in generator.inputs] == [[None, Z], [None, META]]
    z = np.random.default_rng(0).normal(0, 1, (1, Z)).astype("float32")
    a = generator.predict([z, np.full((1, META), -2, "float32")], verbose=0)
    b = generator.predict([z, np.full((1, META), 2, "float32")], verbose=0)
    assert a.shape == (1, SIZE, SIZE, 3)
    assert np.abs(a - b).max() > 1e-4


def test_discriminator_output_depends_on_metadata():
    """#4: the discriminator judges an image against its metadata."""
    discriminator = build_discriminator((SIZE, SIZE, 3), META)
    assert len(discriminator.inputs) == 2
    img = np.zeros((1, SIZE, SIZE, 3), "float32")
    a = discriminator.predict([img, np.full((1, META), -2, "float32")], verbose=0)
    b = discriminator.predict([img, np.full((1, META), 2, "float32")], verbose=0)
    assert a.shape == (1, 1)
    assert abs(float(a[0, 0] - b[0, 0])) > 1e-6


def test_unconditioned_models_still_build():
    generator, discriminator, combined = tiny_models(meta_dim=0)
    assert len(generator.inputs) == 1
    assert len(discriminator.inputs) == 1
    assert combined.output_shape == (None, 1)


def test_train_feeds_metadata_rows_to_every_model(tmp_path, monkeypatch):
    """#4: model.train passes the sampled metadata rows to G, D and combined."""
    monkeypatch.chdir(tmp_path)
    generator, discriminator, combined = tiny_models()
    imgs, metadata = synthetic()
    rows = {tuple(r) for r in metadata}
    seen = {"generator": [], "discriminator": [], "combined": []}

    def spy(name, method):
        def wrapped(x, *args, **kwargs):
            seen[name].append(x)
            return method(x, *args, **kwargs)
        return wrapped

    monkeypatch.setattr(generator, "predict", spy("generator", generator.predict))
    monkeypatch.setattr(discriminator, "fit", spy("discriminator", discriminator.fit))
    monkeypatch.setattr(combined, "fit", spy("combined", combined.fit))

    model.train(generator, discriminator, combined, imgs, metadata,
                epochs=2, batch_size=2, save_interval=10)

    assert len(seen["generator"]) == 2
    assert len(seen["discriminator"]) == 4
    assert len(seen["combined"]) == 2
    for name, calls in seen.items():
        for x in calls:
            assert isinstance(x, list) and len(x) == 2, name
            assert all(tuple(r) in rows for r in np.asarray(x[1])), name


def test_train_rejects_mismatched_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    generator, discriminator, combined = tiny_models()
    imgs, _ = synthetic()
    with pytest.raises(ValueError, match="metadata shape"):
        model.train(generator, discriminator, combined, imgs,
                    np.zeros((len(imgs), META + 1)), 1, 2, 1)


def test_train_disables_tensorboard_histograms(tmp_path, monkeypatch):
    """#7: weight histograms of the full-size generator exhaust memory."""
    monkeypatch.chdir(tmp_path)
    created = []
    real_tensorboard = model.TensorBoard

    def recording_tensorboard(**kwargs):
        created.append(kwargs)
        return real_tensorboard(**kwargs)

    monkeypatch.setattr(model, "TensorBoard", recording_tensorboard)
    generator, discriminator, combined = tiny_models()
    imgs, metadata = synthetic()
    model.train(generator, discriminator, combined, imgs, metadata,
                epochs=1, batch_size=2, save_interval=1)

    assert created and all(kw["histogram_freq"] == 0 for kw in created)
    assert (tmp_path / "generator_model_epoch_0.h5").exists()


def max_delta(before, after):
    return max(float(np.abs(a - b).max()) for a, b in zip(before, after))


def test_discriminator_learns_and_combined_freezes_it():
    """Intended GAN wiring: D.fit still trains D; combined.fit does not."""
    generator, discriminator, combined = tiny_models()
    imgs, metadata = synthetic()
    y = np.ones((len(imgs), 1))

    before = [w.copy() for w in discriminator.get_weights()]
    discriminator.fit([imgs, metadata], y, epochs=1, verbose=0)
    assert max_delta(before, discriminator.get_weights()) > 0

    d_before = [w.copy() for w in discriminator.get_weights()]
    g_before = [w.copy() for w in generator.get_weights()]
    z = np.random.default_rng(1).normal(0, 1, (len(imgs), Z))
    combined.fit([z, metadata], y, epochs=1, verbose=0)
    assert max_delta(d_before, discriminator.get_weights()) == 0
    assert max_delta(g_before, generator.get_weights()) > 0
