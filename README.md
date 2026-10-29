# Metamon Gen 9 OU baseline

A small, inference-only harness around [UT-Austin-RPL/metamon](https://github.com/UT-Austin-RPL/metamon). The first milestone is **a pretrained Kakuna checkpoint completing legal Gen 9 OU battles**, with auditable outcomes, replays and decision latency—not training a new model or replacing Pokémon Showdown.

## Status: prepared, not battle-validated

Cloud preparation on 2026-10-29 verified dependency imports, the simulator build and loopback startup, checkpoint integrity/deserialization, team legality, and unit tests. **No policy inference or battles have been measured yet.** The user's PC was off; local play is deliberately deferred. See [the exact validation receipt](reports/preparation-2026-10-29.json).

Kakuna's original architecture requires CUDA + FlashAttention-2, including a `(96, 0)` attention window. This project **does not silently substitute CPU attention**. A CPU environment is useful for preparation, not exact Kakuna play. CUDA installation and the user's hardware still need local verification.

Important finding: the pinned upstream `competitive` pool contains 16 teams, but five fail the pinned simulator's Gen 9 OU validator because they use Tera Blast. The harness retains the **11 unmodified, valid teams**; it does not weaken the rules or rewrite moves. That selection changes the evaluation population and must be reported.

## Start here tomorrow

Use Linux or Linux inside WSL2, Python **3.12.14**, Node **24.22.1**, Git and [uv](https://docs.astral.sh/uv/getting-started/installation/). For exact inference, use a compatible NVIDIA GPU, driver and CUDA toolkit. See [local setup and troubleshooting](docs/local-run.md).

```bash
git clone https://github.com/haidmoham/metamon-ou-baseline.git
cd metamon-ou-baseline
bash scripts/setup.sh cuda           # use cpu for preparation-only checks
.venv/bin/python scripts/prefetch.py  # ~572 MB checkpoint; no training datasets
.venv/bin/python scripts/validate_teams.py
PYTHONPATH=src .venv/bin/python -m ou_baseline.run --check-env
```

In terminal 1, start the pinned server. It is deliberately local-only and has no public ladder connection:

```bash
bash scripts/start-server.sh
```

In terminal 2, run the first smoke test—**one battle against each reference**, sequentially:

```bash
PYTHONPATH=src .venv/bin/python -m ou_baseline.run \
  --opponents RandomBaseline PokeEnvHeuristic --battles 1 \
  --output results/smoke-001
```

After successful smoke tests, collect a larger baseline:

```bash
PYTHONPATH=src .venv/bin/python -m ou_baseline.run \
  --opponents RandomBaseline PokeEnvHeuristic GymLeader EmeraldKaizo \
  --battles 100 --output results/baseline-001
```

The `--battles` count is **per opponent**, not total. Each output directory must be fresh. These are existing upstream reference agents, not new heuristics. The project makes no claim that they are human-strength opponents.

### Challenge it yourself

Connect to `http://localhost:8000`, choose the exact local username `HumanPlayer`, and prepare a legal **Gen 9 OU** team. Start the acceptor, then challenge `KakunaLocal` in that format:

```bash
PYTHONPATH=src .venv/bin/python -m ou_baseline.run --mode challenge \
  --username KakunaLocal --opponent-username HumanPlayer --role acceptor \
  --battles 1 --output results/human-001
```

This is a local challenge, not a public account, a public ladder run, or an automatic public replay upload. Browser connectivity and human play remain to be checked on the PC.

## What gets recorded

| File | Purpose |
| --- | --- |
| `manifest.json` | Pins, runtime, model hash, legal team hashes, seeds and scope |
| `battles.jsonl` | Per-battle result, turn count, invalid-action count, protocol errors |
| `decisions.jsonl` | Chosen action, legal mask and decision latency |
| `summary.json` | W/L/ties/incomplete, win rate, Wilson 95% interval, median/p95 latency |
| `replays/*.log`, `*.html` | Protocol evidence and upstream replay viewer wrapper |
| `trajectories/`, `error.txt` | Upstream trajectories; diagnostic traceback on failure |

An illegal policy action, invalid server choice or incomplete battle fails the run rather than silently counting a fallback as success. Raw results, caches, weights and replays are Git-ignored. Review/redact before any future publication. HTML replay viewing may fetch public Showdown viewer assets; the log remains available without rendering.

**Measurement limits:** random upstream team-preview lead; stochastic actions at temperature 1; Python/NumPy/Torch seeds do not seed simulator RNG; single-actor latency includes tensor preparation/device synchronization and the first action, but excludes simulator/network wait. Frozen simulator rules are not a claim about today's public ladder. Small samples do not establish strength or Elo/GXE. See [evaluation protocol](docs/evaluation.md).

## Reproducibility and attribution

- Exact upstream Git/Hugging Face revisions and asset SHA-256: [`pins.json`](pins.json).
- Fully versioned Linux/Python dependencies: `requirements-base.lock`, plus CPU or CUDA torch lock. The CUDA/FlashAttention installation has **not** been tested here.
- Simulator dependencies use `npm ci` against the lockfile at the pinned Showdown commit.
- This repository contains the adapter, documentation and tests, **not model weights, team datasets, Pokémon art, or a simulator fork**.
- Upstream source: MIT; Hugging Face model card: Apache-2.0. These licenses do not establish rights in unrelated Pokémon branding or all training data. [Rights and attribution audit](docs/rights-and-attribution.md).

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
```

No training, fine-tuning, spending, credential creation or public ladder automation is part of this baseline.
