"""Save the 3 hardcoded example questionnaires to JSON files."""

import json
from pathlib import Path

from synthpersona.questionnaire.examples import EXAMPLE_QUESTIONNAIRES

OUTPUT_DIR = Path("questionnaires")
OUTPUT_DIR.mkdir(exist_ok=True)

for q in EXAMPLE_QUESTIONNAIRES:
    path = OUTPUT_DIR / f"{q.name}.json"
    path.write_text(json.dumps(q.model_dump(), indent=2))
    print(f"Saved {path} ({len(q.questions)} questions, {len(q.dimensions)} dimensions)")
