from pathlib import Path
import os

from experiments.evaluate_panda_baselines import latest_file


def test_latest_file_selects_newest_checkpoint(tmp_path):
    old = tmp_path / "old.pth"
    new = tmp_path / "new.pth"
    old.write_bytes(b"old")
    new.write_bytes(b"new")
    os.utime(old, (1, 1))
    os.utime(new, (2, 2))
    assert latest_file(tmp_path, "*.pth") == new
