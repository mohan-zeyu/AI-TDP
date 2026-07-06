"""Parser tests: synthetic golden fixtures in both encodings + a real-file smoke test."""

from pathlib import Path

import numpy as np
import pytest

from tdp.io.hm_csv import HMParseError, read_hm_csv

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "实验数据（黑胶带版）"

ROWS, COLS = 6, 8


def _fixture_text(values: np.ndarray) -> str:
    lines = [
        "文件路径:,C:\\Users\\x\\Desktop\\HM20260704180806.csv,,",
        "测温单位,摄氏度,,",
        "全局,,,",
        "参数:,,发射率:,0.93",
        ",,反射温度:,25.0",
        ",,距离:,0.1",
        ",,环境温度:,28.5",
        ",,湿度:,0.0",
        f"统计:,,平均值:,{values.mean():.1f}",
        f",,最小值:,{values.min():.1f}",
        f",,最大值:,{values.max():.1f}",
        ",,,",
        "坐标Y\\X," + ",".join(str(i) for i in range(COLS)),
    ]
    for r in range(ROWS):
        lines.append(f"{r}," + ",".join(f"{v:.1f}" for v in values[r]))
    return "\r\n".join(lines) + "\r\n"


@pytest.fixture
def values() -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.round(rng.uniform(25.0, 48.0, (ROWS, COLS)), 1)


@pytest.mark.parametrize("encoding", ["gbk", "utf-8-sig"])
def test_roundtrip_both_encodings(tmp_path: Path, values: np.ndarray, encoding: str):
    p = tmp_path / "HM20260704180806.csv"
    p.write_bytes(_fixture_text(values).encode(encoding))
    frame = read_hm_csv(p, expect_shape=(ROWS, COLS))
    np.testing.assert_allclose(frame.temps, values.astype(np.float32), atol=1e-4)
    assert frame.meta["emissivity"] == 0.93
    assert frame.meta["reflected_temp_c"] == 25.0
    assert frame.meta["env_temp_c"] == 28.5
    assert frame.timestamp is not None and frame.timestamp.hour == 18
    assert frame.encoding == ("gb18030" if encoding == "gbk" else "utf-8-sig")


def test_self_check_catches_corruption(tmp_path: Path, values: np.ndarray):
    text = _fixture_text(values).replace(f"{values[2, 3]:.1f}", f"{values[2, 3] + 40:.1f}", 1)
    p = tmp_path / "HM20260704180806.csv"
    p.write_bytes(text.encode("gbk"))
    with pytest.raises(HMParseError, match="self-check"):
        read_hm_csv(p, expect_shape=(ROWS, COLS))


def test_wrong_shape_rejected(tmp_path: Path, values: np.ndarray):
    p = tmp_path / "HM20260704180806.csv"
    p.write_bytes(_fixture_text(values).encode("gbk"))
    with pytest.raises(HMParseError, match="shape"):
        read_hm_csv(p, expect_shape=(ROWS + 1, COLS))


@pytest.mark.skipif(not RAW_DIR.exists(), reason="raw measurement data not present")
def test_real_file_smoke():
    sample = sorted((RAW_DIR / "case02_满载" / "data").glob("HM*.csv"))[0]
    frame = read_hm_csv(sample)
    assert frame.temps.shape == (256, 192)
    assert 20.0 < frame.temps.min() < 35.0
    assert 30.0 < frame.temps.max() < 90.0
    assert frame.meta["emissivity"] == 0.93
