#!/usr/bin/env python3
"""Create a new ADR from the standard template.

Usage:
    python scripts/new_adr.py 26 "GEPA over MIPROv2" "gepa-over-miprov2"
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

TEMPLATE = """# ADR-{number}: {title}

**Date:** {date}
**Status:** Proposed

## Context

What is the issue that we're seeing that motivates this decision?

## Decision

What is the change that we're proposing and/or doing?

## Alternatives Considered

What other options were considered? Why were they rejected?

## Pros

What are the advantages of this decision?

## Cons

What are the disadvantages of this decision?

## Risks

What are the risks? How are they mitigated?

## Future Reconsideration Criteria

What would make us revisit this decision?
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("number", type=int, help="ADR number (e.g. 26)")
    parser.add_argument("title", help="ADR title (e.g. 'GEPA over MIPROv2')")
    parser.add_argument("slug", help="URL-safe slug (e.g. 'gepa-over-miprov2')")
    args = parser.parse_args()

    adr_dir = Path("docs/adr")
    adr_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{args.number:04d}-{args.slug}.md"
    path = adr_dir / filename

    if path.exists():
        print(f"ERROR: {path} already exists", file=sys.stderr)
        return 1

    content = TEMPLATE.format(
        number=args.number,
        title=args.title,
        date=date.today().isoformat(),
    )

    path.write_text(content)
    print(f"Created {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
