"""验证暂停播放会取消待执行的回调。"""

import os
import sys
from pathlib import Path
import types
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _support import load_radar_viewer


class PlaybackTimerTest(unittest.TestCase):
    def test_pause_cancels_pending_callback_before_resume(self):
        radar_viewer = load_radar_viewer("radar_viewer_timer_test")

        class FakePlayer:
            toggle = radar_viewer.RadarPlayer.toggle
            loop = radar_viewer.RadarPlayer.loop

            def __init__(self):
                self.fns = [0, 1, 2]
                self.current = 0
                self.playing = True
                self.timer = "pending-timer"
                self.cancelled = []
                self.scheduled = []
                self.speed_var = types.SimpleNamespace(get=lambda: 100)

            def update_frame(self):
                pass

            def after(self, delay, callback):
                token = f"timer-{len(self.scheduled) + 1}"
                self.scheduled.append((token, delay, callback))
                return token

            def after_cancel(self, token):
                self.cancelled.append(token)

        player = FakePlayer()
        player.toggle()
        player.toggle()

        self.assertEqual(player.cancelled, ["pending-timer"])
        self.assertEqual(len(player.scheduled), 1)
        self.assertEqual(player.timer, "timer-1")
        self.assertEqual(player.current, 1)


if __name__ == "__main__":
    unittest.main()
