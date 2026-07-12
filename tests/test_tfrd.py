"""TFRD adapter smoke tests on synthetic .mat fixtures (real data is external)."""

from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
import torch

from tdp.eval.tfrd import AMBIENT_K, TFRDDataset, evaluate, read_list, sensors_from_obs
from tdp.model.operator import ModelConfig, ThermalOperatorV2
from tdp.train.pretrain import collate


@pytest.fixture
def tfrd_root(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    (tmp_path / "train" / "train").mkdir(parents=True)
    names = []
    for i in range(4):
        n = 64
        u = AMBIENT_K + 30 * rng.random((n, n))
        obs = np.zeros_like(u)
        rows, cols = rng.integers(4, n - 4, 5), rng.integers(4, n - 4, 5)
        obs[rows, cols] = u[rows, cols]
        name = f"sample{i}.mat"
        sio.savemat(tmp_path / "train" / "train" / name,
                    {"u": u, "u_obs": obs, "F": np.zeros((n, n))})
        names.append(f"train/{name}")
    (tmp_path / "train" / "train_val.txt").write_text("\n".join(names))
    return tmp_path


def test_read_list_and_sensors(tfrd_root: Path):
    files = read_list(tfrd_root, "train/train_val.txt")
    assert len(files) == 4
    m = sio.loadmat(str(files[0]))
    xy, temps = sensors_from_obs(m["u_obs"])
    assert xy.shape == (5, 2) and temps.min() > AMBIENT_K - 1
    assert xy.max() <= 1.0


def test_dataset_collates_and_model_runs(tfrd_root: Path):
    files = read_list(tfrd_root, "train/train_val.txt")
    ds = TFRDDataset(files, n_query=32)
    batch = collate([ds[i] for i in range(len(ds))])
    sensors, s_mask, q_xy, q_theta, *_rest = batch
    model = ThermalOperatorV2(ModelConfig(d_model=32, n_fourier=8))
    pred = model(sensors, q_xy, None, sensor_mask=s_mask)
    assert pred.shape == q_theta.shape
    m = evaluate(model, files, limit=2)
    assert m["n_samples"] == 2 and np.isfinite(m["mae_K"])
