# First evaluation protocol

## Acceptance gate

The first milestone is reached only when the **actual pinned Kakuna**:

1. Loads checkpoint 34 with its expected architecture, unchanged attention window and verified SHA-256.
2. Completes Gen 9 OU games against `RandomBaseline` and `PokeEnvHeuristic` on the pinned simulator.
3. Has zero illegal choices/fallbacks, zero server choice errors and no incomplete games in that run.
4. Produces inspectable battle protocol, replays, per-decision timing and a manifest.
5. Ideally completes a local challenge against a human; this is a separate outcome, not implied by automated games.

The preparation report deliberately has `completed_battles: 0` and null performance fields. Do not replace them with upstream GXE claims or hypothetical results.

## Comparisons

Start with one battle per opponent for plumbing, then 100 per opponent for an initial descriptive comparison. Random and poke-env's heuristic establish basic references, not a meaningful expert-strength claim. `GymLeader` and `EmeraldKaizo` offer additional existing upstream heuristic references. A later head-to-head against another released pretrained policy (for example Minikazam or Kadabra3) should be a separately pinned experiment, not an undocumented replacement.

Both sides independently sample from the same 11-team, validator-approved pool. Record its hashes. The five invalid upstream teams are excluded without modification; all results must identify that filtered pool. Team-preview uses the upstream random lead; no separate preview checkpoint is loaded. Keep temperature 1.0 for the initial run.

Run sequentially, one actor at a time. The upstream generic heuristic CLI defaults to 30 concurrent actors, which is intentionally not used. Keep hardware, driver, thermal/power conditions, model config and software pins fixed when comparing latency. Separate cold initialization/download time from policy-decision latency; this runner also reports total wall time.

## Statistics and failure accounting

Report W/L/ties/incomplete per opponent; ties are non-wins in the binary win-rate Wilson 95% interval. Incomplete games are excluded from that denominator and reported separately, never turned into losses. Small-N intervals are broad; no Elo, GXE or human percentile can be inferred from these bot matches. Stochastic teams/actions and an unseeded simulator mean reruns are not bit-identical.

Decision latency includes observation tensor preparation, policy inference, GPU synchronization and action transfer. It excludes simulator/network waiting and includes the first action. Median, nearest-rank p95, max and sample count are descriptive; report GPU model, torch/CUDA/FlashAttention versions and actor count alongside them.

The runner checks the legal mask before sending an action and retains protocol errors. It prevents the upstream wrapper's broad reset-on-exception path from silently discarding a failed battle. Unit tests cover reporting and protocol handling with synthetic messages, not real simulator-policy integration.

## Privacy and publication

Raw logs are local/Git-ignored. The protocol tap omits private request payloads and chat, but usernames, chosen teams, replay data and diagnostic paths can still be sensitive. Review all logs and derived reports before publishing. Do not upload a human opponent's messages or private data, model weights, or downloaded team files. Commit only an intentional sanitized result receipt after review; external replay upload is not automatic.
