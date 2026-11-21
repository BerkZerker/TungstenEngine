#!/bin/sh
# Build the game optimised and run it. Arguments are passed on to it:
#   ./start.sh --world test --seed 7
#   ./start.sh --help
set -e
cd "$(dirname "$0")"
exec cargo run --release -p tungsten-game -- "$@"
