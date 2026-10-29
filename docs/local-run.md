# Local runbook

## Hardware gate first

The exact Kakuna configuration selects upstream FlashAttention-2 and bfloat16. The supported initial path here is **Linux/WSL2 + NVIDIA Ampere or newer**. The CPU preparation environment cannot validate CUDA kernels, GPU memory use, inference latency or completed battles. Do not interpret passing unit tests as a battle-ready claim.

Check on the PC:

```bash
nvidia-smi
nvcc --version
node --version
uv --version
```

The intended pinned stack is Python 3.12.14, torch 2.6.0+cu124 and flash-attn 2.7.4.post1. `setup.sh cuda` installs it, limiting build parallelism to two jobs by default. The correct NVIDIA driver and CUDA toolkit must already be installed. The setup script does not install system drivers, create credentials, provision a GPU or incur cloud spending. FlashAttention may require a source build; see its [official installation documentation](https://github.com/Dao-AILab/flash-attention/tree/v2.7.4.post1#installation-and-features).

If the PC has an unsupported GPU, an Apple GPU, or no GPU, **stop at preparation** and choose a separately named compatible upstream checkpoint or a reviewed attention implementation later. Neither is silently substituted here. That would be a different measured configuration.

## Setup scope

Run commands from the repository root. `setup.sh cpu` and `setup.sh cuda` target `.venv`; pick the intended mode deliberately. Downloads live in `.cache` and upstream code in `vendor`, all ignored. Use a fresh checkout rather than overwriting an independently edited upstream checkout. The scripts reject mismatched source revisions and preserve a modified validated team pool.

The lock is an actual Linux x86_64 / CPython 3.12 preparation environment, not a claim of cross-platform binary compatibility. It was installed using uv. Direct dependency Git references are immutable commits. Additional CUDA dependencies are resolved by torch's CUDA distribution and need verification on the target PC.

The exact upstream checkpoint is ~572 MB. Its public Hub LFS SHA-256 is verified before deserialization. Only inference weights and the small competitive team archive are fetched; no replay training corpora are requested.

## Local server safety

`configure_server.py` sets `127.0.0.1:8000` and disables the administrative REPL. `start-server.sh` refuses a non-loopback configuration and supplies an explicit port, preventing upstream `cloud-env` IP autodetection from overriding the loopback address. `--no-security` is acceptable only on this local test server. Do not expose it with a tunnel, port forwarding or a LAN bind.

Expected startup line:

```text
Worker 1 now listening on 127.0.0.1:8000
```

The prep run verified both the TCP connection and `/proc/net/tcp` loopback binding. On your own Linux/WSL environment, also check `ss -ltnp | grep :8000` and ensure no unrelated server already occupies the port. The runner verifies the source checkout, not the identity of an arbitrary process already listening there.

Stop the server with Ctrl-C when finished. No daemon or scheduled job is installed. WSL-to-Windows browser forwarding may need troubleshooting; do not fix it by making the insecure server public.

## If a run fails

- **Missing FlashAttention/CUDA:** `--check-env` is safe on CPU and reports the exact limitation. No CPU policy replacement happens.
- **No/changed team manifest:** rerun `scripts/validate_teams.py`; preserve any independently edited cache before starting fresh. Never remove OU bans to make a team pass.
- **Illegal action, server rejection or parser failure:** keep the run directory and `error.txt`; do not count it as a loss or success. The harness fails closed.
- **Waits indefinitely:** verify server startup, legal team choice, exact challenge username and matching Gen 9 OU format. For a smoke test, use an outer process timeout and preserve the partial output (e.g. GNU `timeout --signal=INT 10m ...`). Interrupting a run does not establish battle completion.
- **Gym warning:** the pinned upstream AMAGO dependency imports legacy Gym, which emits a NumPy 2 warning; the Metamon environment uses Gymnasium. Imports passed, but real battle behavior still requires local verification.
- **Build-cache permissions:** set `UV_CACHE_DIR` and `npm_config_cache` to writable local directories; setup already defaults them inside `.cache`.

Never paste passwords, access tokens or private conversations into logs or the public repository.
