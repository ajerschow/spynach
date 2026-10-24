#!/bin/sh
# Runs oracle/matlab/easyspin/run_easyspin_fixtures.m in MATLAB batch mode.
# Paths to EasySpin, the repository and the fixture output are set in run_easyspin_fixtures.m.
# Override the MATLAB executable with: MATLAB=/path/to/matlab oracle/run_easyspin_fixtures.sh
set -eu

MATLAB="${MATLAB:-/Applications/MATLAB_R2026b.app/bin/matlab}"
SCRIPT="$(cd "$(dirname "$0")/matlab/easyspin" && pwd)/run_easyspin_fixtures.m"

if [ ! -x "$MATLAB" ]; then
    echo "MATLAB executable not found: $MATLAB (set MATLAB=...)" >&2
    exit 1
fi

exec "$MATLAB" -batch "run('$SCRIPT')"
