# synthpersona

> **Unofficial implementation** of ["Persona Generators: Generating Diverse Synthetic Personas at Scale"](https://arxiv.org/abs/2602.03545) (Paglieri et al., 2026, Google DeepMind).

Generate diverse populations of synthetic personas for any context. The pipeline takes a short text description, expands it into a structured questionnaire with diversity axes, generates a population of unique personas using Sobol quasi-random sampling, simulates their responses, and measures population diversity across six metrics. An optional evolutionary loop (inspired by AlphaEvolve) iteratively improves the generator code.

## How It Works

```
Short Description
      |
      v
┌─────────────────────┐
│  Questionnaire Gen  │  LLM expands description into context,
│  (2-step LLM)       │  diversity axes, and Likert-scale items
└─────────┬───────────┘
          v
┌─────────────────────┐
│  Persona Generator  │  Stage 1: Sobol sampling → persona descriptors
│  (2-stage)          │  Stage 2: Parallel expansion → full personas
└─────────┬───────────┘
          v
┌─────────────────────┐
│  Simulation         │  Each persona answers each item via
│  (Logic of          │  "logic of appropriateness" (3 questions)
│   Appropriateness)  │  Memory reset between items
└─────────┬───────────┘
          v
┌─────────────────────┐
│  Diversity Metrics  │  6 metrics: coverage, convex hull volume,
│                     │  min/mean pairwise distance, dispersion,
│                     │  KL divergence
└─────────┬───────────┘
          v
┌─────────────────────┐
│ Evolution (optional)│  AlphaEvolve-style loop: mutate generator
│                     │  code, evaluate, keep per-metric elites
└─────────────────────┘
```

### Two-Stage Persona Generation

The core innovation from the paper. Stage 1 uses **Sobol quasi-random sampling** to pick positions in [0,1] for each diversity axis, then asks the LLM to generate persona descriptors as first-person paragraphs embedding those numeric scores. This controls population-level diversity. Stage 2 expands each descriptor into a full persona in parallel — adding depth without affecting the diversity distribution.

### Simulation via Logic of Appropriateness

Each persona answers every questionnaire item independently by asking three questions (from Concordia/Leibo et al.):

1. *What kind of situation is this?*
2. *What kind of person am I?*
3. *What does a person like me do in a situation like this?*

Memory is reset between questions to prevent carryover effects. Responses are scored on the Likert scale and averaged per dimension to produce a population embedding.

### Diversity Metrics

| Metric | Goal | What it measures |
|--------|------|-----------------|
| Coverage | Maximize | Fraction of space within reach of at least one persona (Monte Carlo estimate) |
| Convex Hull Volume | Maximize | Volume of the smallest convex set containing all personas |
| Min Pairwise Distance | Maximize | Ensures no two personas are near-identical |
| Mean Pairwise Distance | Maximize | Average spread across the population |
| Dispersion | Minimize | Radius of the largest empty ball (gaps in coverage) |
| KL Divergence | Minimize | Divergence from an ideal quasi-random reference distribution |

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Google Cloud credentials with access to Vertex AI (Gemini models)

### Install

```bash
git clone <repo-url>
cd synthetic_persona
uv sync
```

### Configure

Create a `.env` file in the project root:

```env
VERTEXAI_PROJECT=your-gcp-project-id
VERTEXAI_LOCATION=global
```

Authenticate with Google Cloud:

```bash
gcloud auth application-default login
```

## Usage

### Quick Start: Generate Personas in 3 Commands

```bash
# 1. Generate a questionnaire from a topic description
uv run synthpersona generate-questionnaire \
  "Climate anxiety among coastal Australians in 2024" \
  -o questionnaire.json

# 2. Generate 25 diverse personas
uv run synthpersona generate-personas questionnaire.json \
  -n 25 -o personas.json

# 3. Simulate their responses and compute diversity
uv run synthpersona simulate questionnaire.json personas.json \
  -o embeddings.json
```

### Using Built-in Questionnaires

The package includes 3 hardcoded questionnaires from the paper (Appendix A.2):

- **AGI Job Displacement 2035** — reactions to AGI-driven job loss (2 axes)
- **American Conspiracy Theories 2024** — belief in conspiracy theories (3 axes)
- **Elderly Rural Japan 2010** — village life attitudes (3 axes)

Save one to JSON:

```python
import json
from synthpersona.questionnaire.examples import AGI_JOB_DISPLACEMENT

with open("agi.json", "w") as f:
    json.dump(AGI_JOB_DISPLACEMENT.model_dump(), f, indent=2)
```

### CLI Commands

```
synthpersona generate-questionnaire <description> [-o FILE]
    Generate a questionnaire from a short topic description.

synthpersona generate-personas <questionnaire.json> [-n NUM] [-o FILE]
    Generate N personas for a questionnaire. Default N=25.

synthpersona simulate <questionnaire.json> <personas.json> [-o FILE]
    Simulate personas answering questionnaire items.

synthpersona metrics <embeddings.json> -d DIM1 -d DIM2 [-d DIM3]
    Compute 6 diversity metrics on population embeddings.

synthpersona evolve [-i ITERATIONS] [--islands NUM] [-q FILE ...]
    Run the evolutionary optimization loop.
    Uses built-in questionnaires if none provided.
```

### Python API

```python
import asyncio
from synthpersona.config import Settings
from synthpersona.llm import LLMClient
from synthpersona.generator.two_stage import TwoStageGenerator
from synthpersona.simulation.runner import SimulationRunner
from synthpersona.questionnaire.generator import QuestionnaireGenerator

async def main():
    settings = Settings()
    client = LLMClient(settings)

    # Generate a questionnaire
    qgen = QuestionnaireGenerator(client=client, settings=settings)
    questionnaire = await qgen.generate("Viking warriors and Valhalla beliefs")

    # Generate 25 personas
    gen = TwoStageGenerator(client=client, settings=settings)
    personas = await gen.generate(
        questionnaire.context, questionnaire.dimensions, 25
    )

    # Simulate responses
    runner = SimulationRunner(client=client, settings=settings)
    embeddings = await runner.simulate(personas, questionnaire)

    # Compute metrics
    import numpy as np
    from synthpersona.metrics.diversity import compute_all_metrics, normalize_likert

    emb_array = np.array([e.to_array(questionnaire.dimensions) for e in embeddings])
    metrics = compute_all_metrics(normalize_likert(emb_array))
    print(f"Coverage: {metrics.coverage:.3f}")
    print(f"Convex Hull Volume: {metrics.convex_hull_volume:.3f}")

asyncio.run(main())
```

### Running the Evolutionary Loop

The evolution loop mutates the persona generator's Python code using LLM-generated mutations, evaluates each variant on diversity metrics, and keeps per-metric elites across parallel islands:

```bash
# Small test run (5 iterations, 2 islands)
uv run synthpersona evolve -i 5 --islands 2

# Full run with custom questionnaires
uv run synthpersona evolve -i 500 --islands 10 \
  -q questionnaires/climate.json \
  -q questionnaires/silk_road.json
```

## Configuration

All settings can be overridden via environment variables or `.env`:

| Setting | Default | Description |
|---------|---------|-------------|
| `VERTEXAI_PROJECT` | `soe-gemini-llm-agents` | GCP project ID |
| `VERTEXAI_LOCATION` | `global` | Vertex AI region |
| `FAST_MODEL` | `vertex_ai/gemini-3-flash-preview` | Model for persona generation and simulation |
| `SMART_MODEL` | `vertex_ai/gemini-3-pro-preview` | Model for questionnaire generation and mutations |
| `POPULATION_SIZE` | `25` | Default personas per population |
| `MAX_CONCURRENCY` | `10` | Max parallel LLM calls |
| `SIMULATION_TEMPERATURE` | `0.0` | Temperature for simulation (0 = deterministic) |

## Project Structure

```
synthpersona/
├── cli.py                        # Click CLI (5 commands)
├── config.py                     # Settings via pydantic-settings
├── llm.py                        # Async LiteLLM wrapper with retries
├── models/
│   ├── questionnaire.py          # Question, Questionnaire
│   └── persona.py                # PersonaDescriptor, Persona, PopulationEmbedding
├── questionnaire/
│   ├── examples.py               # 3 hardcoded questionnaires (few-shot examples)
│   └── generator.py              # Two-step LLM questionnaire generation
├── generator/
│   ├── base.py                   # PersonaGenerator Protocol
│   ├── prompts.py                # Stage 1 + Stage 2 prompt templates
│   └── two_stage.py              # Sobol sampling → parallel LLM expansion
├── simulation/
│   └── runner.py                 # Logic of appropriateness simulation
├── metrics/
│   └── diversity.py              # 6 metrics + Monte Carlo coverage calibration
└── evolution/
    ├── prompts.py                # System prompt + 25 mutation prompts from paper
    ├── island.py                 # Island model with per-metric elites
    ├── mutator.py                # LLM-based code mutation
    ├── code_loader.py            # Dynamic exec() + validation
    └── loop.py                   # Main evolutionary loop
```

## Development

```bash
uv run ruff check .       # Lint
uv run ruff format .      # Format
uv run ty check           # Type check
```

## Citation

This is an unofficial implementation. If you use this work, please cite the original paper:

```bibtex
@article{paglieri2026persona,
  title={Persona Generators: Generating Diverse Synthetic Personas at Scale},
  author={Paglieri, Davide and Cross, Logan and Cunningham, William A. and Leibo, Joel Z. and Vezhnevets, Alexander Sasha},
  journal={arXiv preprint arXiv:2602.03545},
  year={2026}
}
```
