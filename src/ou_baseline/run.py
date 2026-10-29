"""Inference-only runner: one actor, original Kakuna attention, local server only.

No training call, public ladder connection, account creation or replay publication.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import time
import traceback

from .metrics import require_legal, summarize

MODEL_REVISION = "ac8e88e7da8dc2a2b85176d88f81fd8854b1445b"
TEAMS_REVISION = "bff785c80a06273faee1c675d368948f5dd65d44"
CHECKPOINT = "kakuna/ckpts/policy_weights/policy_epoch_34.pt"
CHECKPOINT_SHA256 = "7e7474d5b2ec6624afdeb29a7cf1dc5c84817d5aee8ec39c1d35a153b18df26c"
SHOWDOWN_COMMIT = "d62d3a39878f7ef1c6e1cfbd84e8feb2ce7a9fdd"


def append_json(path, data):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, sort_keys=True) + "\n")


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check-env", action="store_true", help="Import/runtime preflight only; no downloads, battles, or inference")
    p.add_argument("--mode", choices=["evaluate", "challenge"], default="evaluate")
    p.add_argument("--opponents", nargs="+", default=["RandomBaseline", "PokeEnvHeuristic"])
    p.add_argument("--battles", type=int, default=10, help="Battles PER opponent")
    p.add_argument("--output", type=Path, default=Path("results/local-run"))
    p.add_argument("--seed", type=int, default=20261029)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--username", default="KakunaLocal")
    p.add_argument("--opponent-username")
    p.add_argument("--role", choices=["acceptor", "challenger"], default="acceptor")
    p.add_argument("--teams-dir", type=Path, default=Path(".cache/validated-teams/gen9ou"),
                   help="Validated gen9ou directory with matching ../manifest.json")
    return p


@contextlib.contextmanager
def record_protocol(output):
    """Tap upstream messages without changing parsing, battle decisions or transport.

    Private move requests and chat are excluded. Raw battle logs remain local by default.
    """
    from metamon.env.metamon_player import MetamonPlayer
    original = MetamonPlayer._handle_battle_message
    records = {}
    excluded = {"request", "c", "c:", "chat", "pm", "html", "uhtml", "uhtmlchange", "j", "J", "l", "L", "n"}

    async def wrapped(player, messages):
        tag = re.sub(r"[^a-zA-Z0-9_-]", "", messages[0][0])
        record = records.setdefault(tag, {"lines": [], "errors": [], "result": None})
        for fields in messages[1:]:
            if len(fields) < 2 or fields[1] in excluded:
                continue
            line = "|".join(fields)
            record["lines"].append(line)
            if fields[1] == "error":
                record["errors"].append(line)
            elif fields[1] in {"win", "tie"}:
                record["result"] = fields[1]
        # Write before upstream parsing so a parser exception retains its evidence.
        (output / f"{tag}.log").write_text("\n".join(record["lines"]) + "\n")
        return await original(player, messages)

    MetamonPlayer._handle_battle_message = wrapped
    try:
        yield records
    finally:
        MetamonPlayer._handle_battle_message = original
        # Use the upstream replay viewer template; escape script-closing input.
        from poke_env.data.replay_template import REPLAY_TEMPLATE
        for tag, record in records.items():
            log = "\n".join(record["lines"]).replace("<", "&lt;")
            page = REPLAY_TEMPLATE.replace("{BATTLE_TAG}", html.escape(tag))
            page = page.replace("{PLAYER_USERNAME}", "Agent").replace("{OPPONENT_USERNAME}", "Opponent")
            page = page.replace("{REPLAY_LOG}", f">{tag}\n{log}")
            (output / f"{tag}.html").write_text(page)


def wrap_environment(raw):
    """Use upstream timestep/mask conversion but never its catch-and-reset step."""
    from metamon.rl.metamon_to_amago import MetamonAMAGOWrapper
    from amago.envs import AMAGOEnv

    class StrictWrapper(MetamonAMAGOWrapper):
        step = AMAGOEnv.step

    return StrictWrapper(raw)


def main(argv=None):
    args = parser().parse_args(argv)
    if args.battles < 1 or args.temperature <= 0:
        raise SystemExit("--battles and --temperature must be positive")
    os.environ.setdefault("METAMON_CACHE_DIR", str(Path(".cache/metamon").resolve()))
    os.environ.setdefault("WANDB_MODE", "disabled")
    import numpy as np
    import torch
    import metamon
    import amago
    from metamon.rl.pretrained import Kakuna
    from metamon.env import BattleAgainstBaseline, ChallengeByUsername, TeamSet
    from metamon.baselines import get_baseline
    from huggingface_hub import hf_hub_download

    runtime = {"python": platform.python_version(), "platform": platform.platform(),
               "torch": torch.__version__, "metamon": metamon.__version__,
               "amago": amago.__version__, "poke_env": importlib.metadata.version("poke-env"),
               "cuda_available": torch.cuda.is_available(), "cuda_runtime": torch.version.cuda}
    try:
        import flash_attn
        runtime["flash_attn"] = flash_attn.__version__
    except ImportError:
        runtime["flash_attn"] = None
    if args.check_env:
        runtime["battle_ready"] = bool(runtime["cuda_available"] and runtime["flash_attn"])
        runtime["scope"] = "imports/hardware only; model loading, server, team legality and battles NOT tested"
        print(json.dumps(runtime, indent=2))
        return 0
    if not runtime["cuda_available"] or not runtime["flash_attn"]:
        raise SystemExit("Exact Kakuna requires CUDA + flash-attn. CPU import preflight is available with --check-env. No attention substitution is made.")
    if args.mode == "challenge" and not args.opponent_username:
        raise SystemExit("Challenge mode requires --opponent-username (local server only)")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("Output directory must be empty; choose a new run directory")
    args.output.mkdir(parents=True, exist_ok=True)
    replay_dir = args.output / "replays"
    replay_dir.mkdir()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    manifest = {"runtime": runtime, "model": "Kakuna", "checkpoint": 34,
                "model_revision": MODEL_REVISION, "teams_revision": TEAMS_REVISION,
                "format": "gen9ou", "backend": "metamon", "team_preview": "upstream random lead",
                "seed": args.seed, "seed_scope": "Python/NumPy/Torch only; Showdown RNG not seeded, GPU kernels may be nondeterministic",
                "temperature": args.temperature, "mode": args.mode, "battles_per_opponent": args.battles,
                "attention": "upstream FlashAttention window (96, 0)", "training": False,
                "weights_redistributed": False, "server": "localhost:8000", "status": "starting"}
    write_json(args.output / "manifest.json", manifest)
    battles, latencies = [], []
    started = time.perf_counter()
    error = None
    protocols = {}

    class PinnedKakuna(Kakuna):
        def get_path_to_checkpoint(self, checkpoint):
            if checkpoint != 34:
                raise ValueError("This baseline pins Kakuna checkpoint 34")
            path = hf_hub_download(repo_id=self.HF_REPO_ID, filename=CHECKPOINT,
                                   revision=MODEL_REVISION, cache_dir=self.hf_cache_dir)
            manifest["checkpoint_sha256"] = file_hash(path)
            if manifest["checkpoint_sha256"] != CHECKPOINT_SHA256:
                raise RuntimeError("Checkpoint SHA256 mismatch; refusing to deserialize")
            return path

    try:
        showdown_dir = Path("vendor/metamon/server/pokemon-showdown")
        showdown_head = subprocess.check_output(["git", "-C", str(showdown_dir), "rev-parse", "HEAD"], text=True).strip()
        if showdown_head != SHOWDOWN_COMMIT:
            raise RuntimeError("Showdown checkout is not the pinned revision")
        subprocess.run(["git", "-C", str(showdown_dir), "diff", "--quiet", "HEAD", "--", "sim", "data", "config/formats.ts"], check=True)
        manifest["showdown_commit"] = showdown_head
        manifest["server_identity_note"] = "Verified checkout, not identity of listening process. Start only this checkout on localhost:8000."
        teams_path = args.teams_dir
        validation = json.loads((teams_path.parent / "manifest.json").read_text())
        if (validation.get("format") != "gen9ou" or validation.get("teams_revision") != TEAMS_REVISION
                or validation.get("showdown_commit") != SHOWDOWN_COMMIT):
            raise RuntimeError("Team validation manifest does not match pinned format/revisions")
        teams = TeamSet(str(teams_path), "gen9ou")
        actual_teams = {str(Path(f).relative_to(teams_path)): file_hash(f) for f in teams.team_files}
        if not actual_teams or actual_teams != validation.get("accepted"):
            raise RuntimeError("Validated team set is empty or differs from validation manifest")
        manifest["team_files"] = actual_teams
        manifest["team_validation"] = validation
        model = PinnedKakuna()
        experiment = model.initialize_agent(checkpoint=34, log=False, action_temperature=args.temperature)
        policy = experiment.policy
        policy.eval()
        device = experiment.DEVICE
        write_json(args.output / "manifest.json", manifest)
        opponents = args.opponents if args.mode == "evaluate" else [args.opponent_username]
        with record_protocol(replay_dir) as protocols:
            for opponent in opponents:
                common = dict(battle_format="gen9ou", observation_space=model.observation_space,
                              action_space=model.action_space, reward_function=model.reward_function,
                              battle_backend=model.battle_backend,
                              save_trajectories_to=str(args.output / "trajectories"))
                if args.mode == "evaluate":
                    raw = BattleAgainstBaseline(team_set=teams, opponent_type=get_baseline(opponent), turn_limit=1000, **common)
                else:
                    raw = ChallengeByUsername(player_team_set=teams, player_username=args.username,
                            opponent_username=opponent, num_battles=args.battles, role=args.role, **common)
                env = wrap_environment(raw)
                try:
                    for battle_idx in range(args.battles):
                        tstep, info = env.reset()
                        hidden = policy.init_hidden_state(1, device)
                        decisions = 0
                        while True:
                            torch.cuda.synchronize()
                            tick = time.perf_counter()
                            obs, rl2s, time_idxs = tstep.as_input()
                            obs = {k: torch.from_numpy(v).to(device).unsqueeze(1) for k, v in obs.items()}
                            with torch.inference_mode():
                                actions, hidden = policy.get_actions(obs=obs,
                                    rl2s=torch.from_numpy(rl2s).to(device).unsqueeze(1),
                                    time_idxs=torch.from_numpy(time_idxs).to(device).unsqueeze(1),
                                    hidden_state=hidden, sample=True)
                            action = int(actions.item())
                            torch.cuda.synchronize()
                            elapsed = (time.perf_counter() - tick) * 1000
                            latencies.append(elapsed)
                            require_legal(action, info["legal_actions"])
                            tag = raw.current_battle.battle_tag
                            append_json(args.output / "decisions.jsonl", {"battle": tag, "decision": decisions,
                                        "action": action, "legal_actions": list(map(int, info["legal_actions"])), "latency_ms": elapsed})
                            tstep, _, terminated, truncated, info = env.step(np.array([action], dtype=np.int64))
                            decisions += 1
                            if bool(terminated[0]) or bool(truncated[0]):
                                battle = raw.current_battle
                                protocol = protocols.get(tag, {})
                                if not battle.finished or protocol.get("result") not in {"win", "tie"}:
                                    outcome = "incomplete"
                                elif protocol.get("result") == "tie":
                                    outcome = "tie"
                                else:
                                    outcome = "win" if battle.won else "loss"
                                row = {"opponent": opponent, "battle": tag, "index": battle_idx,
                                       "outcome": outcome, "decisions": decisions, "turns": battle.turn,
                                       "gym_terminated": bool(terminated[0]), "gym_truncated": bool(truncated[0]),
                                       "valid_actions": int(info.get("valid_action_count", 0)),
                                       "invalid_actions": int(info.get("invalid_action_count", 0)),
                                       "server_errors": protocol.get("errors", []), "replay": f"replays/{tag}.html"}
                                battles.append(row)
                                append_json(args.output / "battles.jsonl", row)
                                if row["invalid_actions"] or row["server_errors"] or outcome == "incomplete":
                                    raise RuntimeError("Battle failed legal/completion gate; see battles.jsonl")
                                break
                finally:
                    raw.close()
    except (Exception, KeyboardInterrupt) as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        (args.output / "error.txt").write_text(traceback.format_exc())
    finally:
        summary = summarize(battles, latencies)
        summary.update({"error": error, "wall_seconds": time.perf_counter() - started,
                        "completed_battles": sum(b["outcome"] != "incomplete" for b in battles), "recorded_battles": len(battles), "status": "failed" if error else "completed",
                        "server_error_count": sum(len(r["errors"]) for r in protocols.values())})
        manifest["status"] = summary["status"]
        write_json(args.output / "manifest.json", manifest)
        write_json(args.output / "summary.json", summary)
        print(json.dumps(summary, indent=2))
    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
