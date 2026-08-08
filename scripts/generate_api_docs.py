#!/usr/bin/env python3
"""Generate API documentation from the OpenAPI schema.

Produces:
- docs/api/openapi.json — raw OpenAPI 3.1 spec
- docs/api/README.md — human-readable summary
- docs/api/endpoints.md — per-endpoint reference
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ik_kernel.app import create_app


def main() -> int:
    app = create_app()
    schema = app.openapi()

    out_dir = Path("docs/api")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Raw OpenAPI
    (out_dir / "openapi.json").write_text(json.dumps(schema, indent=2))

    # README
    readme = [
        "# Indus Kernel API",
        "",
        f"Version: {schema.get('info', {}).get('version', 'unknown')}",
        "",
        f"Auto-generated from the FastAPI app. Total endpoints: {sum(len(v) for v in schema.get('paths', {}).values())}",
        "",
        "## Endpoints",
        "",
    ]
    for path, methods in sorted(schema.get("paths", {}).items()):
        for method, info in methods.items():
            if method.startswith("_"):
                continue
            summary = info.get("summary", "")
            tags = info.get("tags", [])
            readme.append(f"### `{method.upper()} {path}`")
            if summary:
                readme.append(f"_{summary}_")
            if tags:
                readme.append(f"Tags: `{', '.join(tags)}`")
            readme.append("")
    (out_dir / "README.md").write_text("\n".join(readme))

    # Endpoints reference
    endpoints = ["# Endpoints", ""]
    for path, methods in sorted(schema.get("paths", {}).items()):
        for method, info in methods.items():
            if method.startswith("_"):
                continue
            endpoints.append(f"## `{method.upper()} {path}`")
            if info.get("summary"):
                endpoints.append(f"**{info['summary']}**")
            if info.get("description"):
                endpoints.append(info["description"])
            if info.get("requestBody"):
                endpoints.append("### Request Body")
                endpoints.append("```json")
                endpoints.append(json.dumps(info["requestBody"], indent=2))
                endpoints.append("```")
            if info.get("responses"):
                endpoints.append("### Responses")
                for code, resp in info["responses"].items():
                    endpoints.append(f"- **{code}**: {resp.get('description', '')}")
            endpoints.append("")
    (out_dir / "endpoints.md").write_text("\n".join(endpoints))

    print(f"Generated {out_dir}/openapi.json, README.md, endpoints.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
