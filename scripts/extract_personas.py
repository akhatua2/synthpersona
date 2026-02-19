"""Extract a clean persona list from eval results, stripping numeric scores."""

import json
import sys
from pathlib import Path


def extract(eval_path: str, output_path: str | None = None) -> None:
    data = json.loads(Path(eval_path).read_text())

    personas = []
    for p in data["personas"]:
        personas.append({
            "name": p["name"],
            "high_level_description": p["descriptor"]["high_level_description"],
            "full_description": p["full_description"],
        })

    result = {
        "questionnaire": data["questionnaire"],
        "dimensions": data["dimensions"],
        "num_personas": data["num_personas"],
        "personas": personas,
    }

    out = Path(output_path) if output_path else Path(eval_path).with_suffix(".clean.json")
    out.write_text(json.dumps(result, indent=2))
    print(f"Wrote {len(personas)} personas to {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_personas.py <eval_results.json> [output.json]")
        sys.exit(1)
    extract(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
