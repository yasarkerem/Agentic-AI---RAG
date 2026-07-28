"""Run the experiment matrix (systems x benchmark tasks) and persist results.

Arrives in Phase 6. This placeholder exists so `make experiment` is honest:
it explains itself and exits non-zero instead of silently succeeding.
"""

import sys

PHASE = 6
NAME = "run_experiment"

if __name__ == "__main__":
    print(
        f"{NAME}: not implemented yet — this is a Phase {PHASE} feature "
        f"(Phase 1 covers the repository foundation only). See WORKLOG.md.",
        file=sys.stderr,
    )
    sys.exit(2)
