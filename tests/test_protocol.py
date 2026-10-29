"""Synthetic protocol-capture tests; these do NOT run or claim real battles."""
import asyncio
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ou_baseline.run import record_protocol

@unittest.skipUnless(importlib.util.find_spec("metamon"), "Metamon not installed")
class ProtocolTests(unittest.TestCase):
    def test_capture_excludes_private_requests_and_chat(self):
        os.environ.setdefault("METAMON_CACHE_DIR", str(Path(".cache/metamon").resolve()))
        from metamon.env.metamon_player import MetamonPlayer
        called = []
        async def original(player, messages):
            called.append(messages)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            with patch.object(MetamonPlayer, "_handle_battle_message", original):
                with record_protocol(target) as records:
                    messages = [[">battle-gen9ou-test"], ["", "init", "battle"],
                                ["", "request", "PRIVATE_TEAM"], ["", "c", "human", "PRIVATE_CHAT"],
                                ["", "turn", "1"], ["", "win", "agent"]]
                    asyncio.run(MetamonPlayer._handle_battle_message(None, messages))
                    self.assertEqual(records["battle-gen9ou-test"]["result"], "win")
                self.assertIs(MetamonPlayer._handle_battle_message, original)
            log = (target / "battle-gen9ou-test.log").read_text()
            self.assertIn("|turn|1", log)
            self.assertNotIn("PRIVATE", log)
            self.assertEqual(len(called), 1)
            self.assertTrue((target / "battle-gen9ou-test.html").exists())

    def test_parser_exception_retains_error_evidence(self):
        from metamon.env.metamon_player import MetamonPlayer
        async def original(player, messages):
            raise RuntimeError("synthetic parser failure")
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(MetamonPlayer, "_handle_battle_message", original):
                with self.assertRaisesRegex(RuntimeError, "synthetic"):
                    with record_protocol(Path(tmp)) as records:
                        asyncio.run(MetamonPlayer._handle_battle_message(None,
                            [[">battle-gen9ou-error"], ["", "error", "[Invalid choice] synthetic"]]))
            self.assertEqual(len(records["battle-gen9ou-error"]["errors"]), 1)
            self.assertTrue((Path(tmp) / "battle-gen9ou-error.log").exists())

if __name__ == "__main__": unittest.main()
