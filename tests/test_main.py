import argparse
import sys

import pytest

from main import parse_args, video_time


@pytest.mark.parametrize(("text", "seconds"), [("750", 750), ("12:30", 750), ("1:02:30", 3750),
                                               ("0:07.5", 7.5)])  # fmt: skip
def test_video_time_formats(text: str, seconds: float) -> None:
    assert video_time(text) == seconds


@pytest.mark.parametrize("text", ["", "abc", "1:2:3:4", "-5", "12:-1"])
def test_video_time_rejects_garbage(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        video_time(text)


def test_start_and_speed_need_a_video(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["main.py", "--video", "p.mp4", "--start", "12:30",
                                      "--speed", "2"])  # fmt: skip
    args = parse_args()
    assert (args.start, args.speed) == (750, 2)
    for extra in (["--start", "10"], ["--speed", "2"], ["--video", "p.mp4", "--speed", "0"]):
        monkeypatch.setattr(sys, "argv", ["main.py", *extra])
        with pytest.raises(SystemExit):
            parse_args()
