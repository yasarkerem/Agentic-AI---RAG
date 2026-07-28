"""Build the machine-readable benchmark file (25 tasks across 5 levels).

Arrives in Phase 5. This placeholder exists so `make benchmark` is honest:
it explains itself and exits non-zero instead of silently succeeding.
"""

import sys

PHASE = 5
NAME = "build_benchmark"

if __name__ == "__main__":
    print(
        f"{NAME}: not implemented yet — this is a Phase {PHASE} feature "
        f"(Phase 1 covers the repository foundation only). See WORKLOG.md.",
        file=sys.stderr,
    )
    sys.exit(2)
