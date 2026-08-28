#!/bin/sh
# Launch the aasys simulation in the interactive 3D view.
#
#   ./run.sh                    default scenario (single, windowed)
#   ./run.sh hovering           pick a scenario by name
#   ./run.sh swarm --headless --duration 60
#   ./run.sh --list             list scenarios, then exit

set -u

cd "$(dirname "$0")" || exit 1
PY="$PWD/.venv/bin/python"

if [ ! -x "$PY" ]; then
    echo "venv python not found at $PY" >&2
    exit 1
fi

if [ "$#" -gt 0 ]; then
    case "$1" in
        -*) exec "$PY" run.py "$@" ;;
        *)  SCENARIO="$1"; shift
            exec "$PY" run.py --scenario "$SCENARIO" "$@" ;;
    esac
else
    exec "$PY" run.py --scenario single
fi