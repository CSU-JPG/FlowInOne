#!/bin/bash
set -euo pipefail

##############################################################################
# FlowInOne Single Test Runner
#
# Runs a single test case by ID from test_configs.json.
#
# Usage:
#   bash tests/run_single_test.sh carnival_01_steampunk
#   bash tests/run_single_test.sh kirby_04_t2i --cfg 9.0 --steps 75
#   bash tests/run_single_test.sh dinner_02_cat --fast
#
# List available test IDs:
#   bash tests/run_single_test.sh --list
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEST_CONFIG="${REPO_ROOT}/tests/test_configs.json"

# Handle --list
if [[ "${1:-}" == "--list" ]]; then
    echo "Available test cases:"
    python3 -c "
import json
with open('${TEST_CONFIG}') as f:
    cases = json.load(f)['test_cases']
for c in cases:
    src = c.get('source_image') or 'none'
    cross = 'skip' if c.get('skip_cross_atten', False) else 'use'
    print(f\"  {c['id']:<30s} [{c['task_type']:<15s}] cross_atten={cross:<4s} src={src}\")
    print(f\"    {c['instruction'][:80]}\")
"
    exit 0
fi

# Get test ID
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <test_case_id> [--cfg N] [--steps N] [--fast] [--gpu N]"
    echo "       $0 --list"
    exit 1
fi

TEST_ID="$1"
shift

# Validate test ID exists
python3 -c "
import json, sys
with open('${TEST_CONFIG}') as f:
    cases = json.load(f)['test_cases']
ids = [c['id'] for c in cases]
if '${TEST_ID}' not in ids:
    print(f'ERROR: Test ID \"${TEST_ID}\" not found.')
    print(f'Available: {\", \".join(ids)}')
    sys.exit(1)
" || exit 1

# Forward remaining args to run_tests.sh
echo "Running single test: ${TEST_ID}"
echo ""

exec bash "${SCRIPT_DIR}/run_tests.sh" --filter "${TEST_ID}" "$@"
