"""Fetch only pinned inference assets. Never download training datasets."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("METAMON_CACHE_DIR", str(ROOT / ".cache/metamon"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teams-only", action="store_true")
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download
    pins = json.loads((ROOT / "pins.json").read_text())
    receipt = {}
    for name in (["teams"] if args.teams_only else ["model", "teams"]):
        spec = pins[name]
        path = Path(hf_hub_download(
            repo_id=spec["repo_id"], filename=spec["filename"], revision=spec["revision"],
            repo_type="dataset" if name == "teams" else "model",
            cache_dir=str(Path(os.environ["METAMON_CACHE_DIR"]) / "pretrained_models") if name == "model" else None,
        ))
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        expected = spec.get("sha256")
        if expected and digest != expected:
            raise RuntimeError(f"{name}: SHA-256 mismatch; refusing this artifact")
        receipt[name] = {"repo_id": spec["repo_id"], "revision": spec["revision"], "filename": spec["filename"], "bytes": path.stat().st_size, "sha256": digest}
        print(f"Verified {name}: {digest}", flush=True)
    out = ROOT / ".cache/download-receipt.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Local receipt: {out.relative_to(ROOT)}. Assets remain ignored, never committed.")


if __name__ == "__main__":
    main()
