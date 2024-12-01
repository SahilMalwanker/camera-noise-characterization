from pathlib import Path

import numpy as np
import pytest

from camera_noise_characterization.cli import main
from camera_noise_characterization.io import load_gray, save_gray
from camera_noise_characterization.sensor import SensorModel, SimulatedCamera

DATA = Path(__file__).resolve().parents[1] / "data"


def test_optics_command_prints_depth_of_field(capsys):
    assert main(["optics"]) == 0
    out = capsys.readouterr().out
    assert "depth of field total [mm]" in out
    assert "336.97" in out
    assert "347.18" in out


def test_fit_command_on_logged_gain_sweep(capsys):
    main(["fit", str(DATA / "lab_gain_sweep.csv")])
    out = capsys.readouterr().out
    assert "K1" in out
    assert "0.021" in out


def test_simulate_then_fit_round_trip(tmp_path, capsys):
    csv_path = tmp_path / "ptc.csv"
    main(["simulate", "ptc", "--steps", "12", "--out", str(csv_path)])
    main(["fit", str(csv_path), "--model", "ptc", "--max-mean", "230"])
    out = capsys.readouterr().out
    assert "conversion gain K" in out
    assert "0.02" in out


def test_image_commands(tmp_path, capsys):
    cam = SimulatedCamera(SensorModel(width=96, height=64, seed=1), seed=1)
    cam.exposure_us = 5000.0
    paths = []
    for name in ("a.png", "b.png"):
        path = tmp_path / name
        save_gray(path, cam.grab())
        paths.append(str(path))
    np.testing.assert_array_equal(load_gray(paths[0]).shape, (64, 96))
    main(["pair", *paths])
    main(["denoise", *paths])
    main(["edges", paths[0], "--method", "sobel", "--out", str(tmp_path / "e.png")])
    out = capsys.readouterr().out
    assert "temporal noise per frame" in out
    assert "binomial 7x7" in out
    assert (tmp_path / "e.png").exists()


def test_bad_inputs_exit_with_a_message_not_a_traceback(tmp_path, capsys):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("gain,exposure_us\n1,100\n", encoding="utf-8")
    for argv in (
        ["fit", str(bad_csv)],
        ["pair", str(tmp_path / "missing_a.png"), str(tmp_path / "missing_b.png")],
        ["edges", str(DATA / "coins_txg08.png"), "--non-edge-fraction", "1.5"],
    ):
        with pytest.raises(SystemExit) as exit_info:
            main(argv)
        assert exit_info.value.code == 2
    assert "missing column" in capsys.readouterr().err
