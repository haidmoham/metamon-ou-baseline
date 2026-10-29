"""Synthetic adapter tests against actual AMAGO wrappers, not simulated battles."""
import importlib.util
import os
from pathlib import Path
import unittest
from ou_baseline.run import wrap_environment

@unittest.skipUnless(importlib.util.find_spec("amago"), "AMAGO not installed")
class AdapterTests(unittest.TestCase):
    def make_env(self, fail=False):
        os.environ.setdefault("METAMON_CACHE_DIR", str(Path(".cache/metamon").resolve()))
        import gymnasium as gym
        import numpy as np
        class Raw(gym.Env):
            action_space = gym.spaces.Discrete(13)
            observation_space = gym.spaces.Dict({"x": gym.spaces.Box(0, 1, (2,), dtype=np.float32)})
            metamon_action_space = None
            metamon_battle_format = "gen9ou"
            metamon_opponent_name = "synthetic"
            resets = 0
            def reset(self, **kwargs):
                self.resets += 1
                return {"x": np.array([0, 1], dtype=np.float32)}, {"legal_actions": [0, 2, 12]}
            def step(self, action):
                self.last_action = action
                if fail:
                    raise RuntimeError("synthetic step failure")
                return {"x": np.array([1, 0], dtype=np.float32)}, 5.0, True, False, {"legal_actions": []}
        raw = Raw()
        return raw, wrap_environment(raw)

    def test_shapes_masks_action_scalar_and_reset(self):
        import numpy as np
        import torch
        raw, env = self.make_env()
        timestep, info = env.reset()
        obs, rl2s, time_idxs = timestep.as_input()
        self.assertEqual(obs["x"].shape, (1, 2))
        self.assertEqual(rl2s.shape, (1, 14))
        self.assertEqual(time_idxs.shape, (1, 1))
        self.assertEqual(torch.from_numpy(time_idxs).unsqueeze(1).shape, (1, 1, 1))
        self.assertEqual(np.flatnonzero(~obs["illegal_actions"][0]).tolist(), [0, 2, 12])
        timestep, reward, terminated, truncated, info = env.step(np.array([12], dtype=np.int64))
        self.assertIsInstance(raw.last_action, int)
        self.assertEqual(raw.last_action, 12)
        self.assertTrue(terminated[0])
        obs, rl2s, time_idxs = timestep.as_input()
        self.assertEqual(rl2s[0, 0], 5)
        self.assertEqual(rl2s[0, 13], 1)
        self.assertEqual(time_idxs[0, 0], 1)
        timestep, _ = env.reset()
        _, rl2s, time_idxs = timestep.as_input()
        self.assertEqual(float(rl2s.sum()), 0)
        self.assertEqual(time_idxs[0, 0], 0)

    def test_exception_is_not_silently_reset(self):
        import numpy as np
        raw, env = self.make_env(fail=True)
        env.reset()
        with self.assertRaisesRegex(RuntimeError, "synthetic step failure"):
            env.step(np.array([0], dtype=np.int64))
        self.assertEqual(raw.resets, 1)

if __name__ == "__main__": unittest.main()
