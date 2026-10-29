#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../vendor/metamon/server/pokemon-showdown"
node -e 'const c = require("./config/config.js"); if(c.bindaddress !== "127.0.0.1" || c.port !== 8000) { console.error("Refusing insecure non-loopback server; run scripts/configure_server.py"); process.exit(1); }'
# An explicit port disables upstream cloud-env IP autodetection, which can
# otherwise override config.bindaddress with 0.0.0.0.
exec node pokemon-showdown start 8000 --no-security
