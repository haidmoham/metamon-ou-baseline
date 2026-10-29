"""Restrict the dedicated local Showdown server to loopback before --no-security."""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
path = root / "vendor/metamon/server/pokemon-showdown/config/config.js"
if not path.exists():
    path.write_text(path.with_name("config-example.js").read_text())
text = path.read_text()
for key, value in (("bindaddress", "'127.0.0.1'"), ("port", "8000"), ("repl", "false")):
    pattern = rf"^exports\.{key}\s*=.*?;.*$"
    replacement = f"exports.{key} = {value};"
    if re.search(pattern, text, re.M):
        text = re.sub(pattern, replacement, text, flags=re.M)
    else:
        text += "\n" + replacement + "\n"
path.write_text(text)
print("Configured Showdown: 127.0.0.1:8000 (not LAN/public).")
