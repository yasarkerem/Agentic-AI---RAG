"""Ingest the PDF/Markdown corpus into chunks and a vector index.

Arrives in Phase 2. This placeholder exists so `make ingest` is honest:
it explains itself and exits non-zero instead of silently succeeding.
"""

import sys

PHASE = 2
NAME = "ingest_corpus"

if __name__ == "__main__":
    print(
        f"{NAME}: not implemented yet — this is a Phase {PHASE} feature "
        f"(Phase 1 covers the repository foundation only). See WORKLOG.md.",
        file=sys.stderr,
    )
    sys.exit(2)
