"""Download pinned teams, retain only teams legal under pinned gen9ou rules."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def main():
    from huggingface_hub import hf_hub_download
    pins = json.loads((ROOT / "pins.json").read_text())
    server = ROOT / "vendor/metamon/server/pokemon-showdown"
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=server, text=True).strip()
    if actual != pins["showdown"]["commit"]:
        raise RuntimeError("Showdown revision differs from pins.json")
    spec = pins["teams"]
    archive = Path(hf_hub_download(repo_id=spec["repo_id"], repo_type="dataset", filename=spec["filename"], revision=spec["revision"]))
    with archive.open("rb") as handle:
        archive_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    if archive_sha != spec["sha256"]:
        raise RuntimeError("Team archive SHA-256 mismatch")
    destination = ROOT / ".cache/validated-teams"
    destination.mkdir(parents=True, exist_ok=True)
    team_dir = destination / "gen9ou"
    # Never overwrite an existing human-edited team pool. Same verified result
    # can be reused; changed pins need a fresh project-local cache directory.
    if team_dir.exists() and not (destination / "manifest.json").exists():
        raise RuntimeError("Existing team directory lacks manifest; preserve it and choose a fresh cache")
    accepted = {}
    rejected = []
    with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as tmp:
        with tarfile.open(archive) as tar:
            tar.extractall(tmp, filter="data")
        candidates = sorted(Path(tmp).rglob("*.gen9ou_team"))
        if not candidates:
            raise RuntimeError("Pinned archive contains no Gen 9 OU teams")
        for path in candidates:
            data = path.read_bytes()
            result = subprocess.run(["node", "pokemon-showdown", "validate-team", "gen9ou"], cwd=server, input=data, capture_output=True, timeout=30)
            if result.returncode:
                rejected.append({"team": path.name, "errors": result.stderr.decode().strip()})
            else:
                accepted[path.name] = hashlib.sha256(data).hexdigest()
        if not accepted:
            raise RuntimeError("No legal teams remain; do not change the simulator rules")
        if team_dir.exists():
            existing = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in team_dir.glob("*.gen9ou_team")}
            if existing != accepted:
                raise RuntimeError("Existing validated pool differs; preserve it and use a fresh cache")
        else:
            team_dir.mkdir()
            for path in candidates:
                if path.name in accepted:
                    shutil.copyfile(path, team_dir / path.name)
    manifest = {"format": "gen9ou", "showdown_commit": actual, "teams_revision": spec["revision"], "source_archive_sha256": archive_sha, "accepted": accepted, "rejected": rejected}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Validated {len(candidates)} teams: {len(accepted)} accepted, {len(rejected)} rejected.")
    for item in rejected:
        print(f"Rejected {item['team']}: {item['errors']}")
    print("No team moves, EVs or simulator rules were modified.")


if __name__ == "__main__":
    main()
