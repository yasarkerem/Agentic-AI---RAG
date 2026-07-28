"""Generate comparison tables, figures, and statistical analysis.

Arrives in Phase 7. This placeholder exists so `make report` is honest:
it explains itself and exits non-zero instead of silently succeeding.
"""

import sys

PHASE = 7
NAME = "generate_report"

if __name__ == "__main__":
    print(
        f"{NAME}: not implemented yet — this is a Phase {PHASE} feature "
        f"(Phase 1 covers the repository foundation only). See WORKLOG.md.",
        file=sys.stderr,
    )
    sys.exit(2)
