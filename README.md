# synthpersona

> **Unofficial implementation** of ["Persona Generators: Generating Diverse Synthetic Personas at Scale"](https://arxiv.org/abs/2602.03545) (Paglieri et al., 2026, Google DeepMind).

Generate diverse populations of synthetic personas for any context. Give it a short text description, and it produces a population of unique personas spanning the full range of relevant attitudes and behaviors — not just stereotypical LLM outputs.

## Paper Architecture

The diagram below shows the **full system from the paper** (Figure 1). Boxes marked with **[WE USE]** are what this implementation provides out of the box. The **AlphaEvolve** feedback loop is implemented but optional — the paper's evolved generators have not been publicly released.

```
                                   ┌───────────────┐
                                   │  AlphaEvolve  │◄──────────────────────────────┐
                                   │  (mutate      │                               │
                                   │   generator   │                               │
                                   │   source code)│                               │
                                   └──────┬────────┘                               │
                                          │ mutated Python code                    │
                  [IMPLEMENTED,           │                                        │
                   OPTIONAL]              v                                        │
                               ╔══════════════════════╗                            │
  ┌──────────────────┐         ║  Persona Generator   ║     ┌──────────────────┐   │
  │  Questionnaires  │         ║  G_φ,θ (c, D, N)     ║────>│  Population of   │   │
  │  q ~ Q           │         ║                      ║     │  N = 25 Personas │   │
  │                  │         ║  Stage 1: Sobol +    ║     │  P               │   │
  │ ┌──────────────┐ │ context ║   descriptors        ║     │                  │   │
  │ │  Context  c  │─┼────────>║  Stage 2: parallel   ║     │ Full text desc.  │   │
  │ ├──────────────┤ │  dims   ║   expansion          ║     │ for each persona │   │
  │ │ Dimensions D │─┼────────>║                      ║     └────────┬─────────┘   │
  │ ├──────────────┤ │         ╚══════════════════════╝              │             │
  │ │ Questions  I │─┼──┐          [WE USE]                          │             │
  │ └──────────────┘ │  │                                            │             │
  └──────────────────┘  │                                            v             │
      [WE USE]          │     ┌──────────────────────────────────────────────┐     │
   50 questionnaires    │     │  Simulation  Z = Ψ(P, I)                     │     │
   included             └────>│                                              │     │
                              │  For each persona × question:                │     │
                              │   1. "What kind of situation is this?"       │     │
                              │   2. "What kind of person am I?"             │     │
                              │   3. "What does a person like me do here?"   │     │
                              │                                              │     │
                              │  Paper: Concordia library                    │     │
                              │  Ours:  Direct LLM calls (same 3 questions)  │     │
                              └──────────────────┬───────────────────────────┘     │
                                  [WE USE]       │                                 │
                                                 │  score vectors Z                │
                                                 │  (N × K matrix)                 │
                                                 v                                 │
                              ┌──────────────────────────────────────┐             │
                              │  Diversity Metrics  M(Z)             │             │
                              │                                      │             │
                              │  coverage, convex hull volume,       │             │
                              │  min/mean pairwise distance,         │─────────────┘
                              │  dispersion, KL divergence           │  fitness signal
                              │                                      │  (only if running
                              │  All 6 metrics from the paper        │   evolution)
                              └──────────────────────────────────────┘
                                  [WE USE]
```

### What we implement vs the paper

| Component | Paper | This Implementation | Status |
|-----------|-------|-------------------|--------|
| Questionnaires | 50 (30 train / 10 val / 10 test) | All 50 included in `questionnaires/` | Full |
| Persona Generator | Evolved via AlphaEvolve | Baseline Sobol seed generator | Baseline only |
| Simulation | Concordia library + gemma-3-27b-it | Direct LLM calls + Gemini Flash | Equivalent logic |
| Diversity Metrics | 6 metrics + calibration | All 6 metrics + calibration | Full |
| AlphaEvolve Loop | 500 iterations, 10 islands | Implemented, optional | Full |

The **only gap** is that we start from the baseline generator (Sobol + two-stage LLM), not the evolved one. The paper hasn't released their evolved generators yet. Run `synthpersona evolve` to evolve your own.

## How It Works

The inference pipeline (what you run day-to-day) has four steps:

### Step 1: Questionnaire Generation

You provide a short description like *"Climate anxiety among coastal Australians in 2024"*. The LLM expands this in two steps:
1. **Expand** the description into a detailed survey context + 2-3 diversity axes (e.g., `existential_eco_dread`, `proactive_adaptation_urgency`, `institutional_betrayal_perception`)
2. **Generate** 3-5 Likert-scale items per axis, including reverse-coded items

Four well-known psychometric instruments (BFI, DASS, SVO, NFCS) and three example questionnaires from the paper serve as few-shot references.

### Step 2: Two-Stage Persona Generation

**Stage 1 (Autoregressive):** Sobol quasi-random sampling picks positions in [0,1] for each diversity axis, then the LLM generates persona descriptors as first-person paragraphs embedding those numeric scores. This controls population-level diversity — ensuring the full space is covered, not just the stereotypical center.

**Stage 2 (Parallel):** Each descriptor is expanded into a full persona independently via `asyncio.gather`. This adds depth (background, values, decision-making style) without affecting the diversity distribution set in Stage 1.

### Step 3: Simulation via Logic of Appropriateness

Each persona answers every questionnaire item independently. For each item, the LLM role-plays as the persona and reasons through three questions (from [Leibo et al., 2024](https://arxiv.org/abs/2412.19010)):

1. *What kind of situation is this?*
2. *What kind of person am I?*
3. *What does a person like me do in a situation like this?*

Memory is reset between items to prevent carryover effects. Responses are scored on the Likert scale and averaged per dimension to produce a score vector for each persona.

### Step 4: Diversity Metrics

Six metrics measure how well the population covers the space:

| Metric | Goal | What it measures |
|--------|------|-----------------|
| Coverage | Maximize | Fraction of space within reach of at least one persona (Monte Carlo) |
| Convex Hull Volume | Maximize | Volume of the smallest convex set containing all personas |
| Min Pairwise Distance | Maximize | Ensures no two personas are near-identical |
| Mean Pairwise Distance | Maximize | Average spread across the population |
| Dispersion | Minimize | Radius of the largest empty ball (gaps in coverage) |
| KL Divergence | Minimize | Divergence from an ideal quasi-random reference distribution |

## Baseline vs Evolved Generators

**This implementation ships with the baseline generator** — the Sobol quasi-random sampling seed (one of three starting generators described in the paper). This is not the final evolved generator from the paper.

The paper's main contribution is an AlphaEvolve-style evolutionary loop that **mutates the generator's Python source code** over 500 iterations to discover generators that produce more diverse populations. The evolved generators substantially outperform all baselines. However, the authors have not yet released the evolved generator code (*"we plan on releasing the full code for the best Persona Generators upon acceptance"*).

**What the paper found the best evolved generators do differently:**
- Produce first-person paragraphs with explicit numeric axis scores (e.g., *"My Threat Appraisal is 0.91"*)
- Replace Stage 2 background paragraphs with appropriateness rules, core beliefs, or inner monologues
- Sobol sampling in Stage 1 survived evolution; formative-memory generators were eliminated

**You have two options:**
1. **Use the baseline as-is** — it already uses Sobol sampling and produces reasonable diversity
2. **Run evolution yourself** via `synthpersona evolve` — this replicates the paper's optimization loop to discover better generators (costs ~500 Gemini Pro calls + evaluations)

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

### Quick Start

```bash
# 1. Generate a questionnaire from a topic description
uv run synthpersona generate-questionnaire \
  "Climate anxiety among coastal Australians in 2024" \
  -o questionnaire.json

# 2. Generate 25 diverse personas
uv run synthpersona generate-personas questionnaire.json \
  -n 25 -o personas.json

# 3. Simulate their responses
uv run synthpersona simulate questionnaire.json personas.json \
  -o embeddings.json

# 4. Compute diversity metrics
uv run synthpersona metrics embeddings.json \
  -d existential_eco_dread \
  -d proactive_adaptation_urgency \
  -d institutional_betrayal_perception
```

### CLI Reference

| Command | Description |
|---------|-------------|
| `synthpersona generate-questionnaire <description> [-o FILE]` | Generate a questionnaire from a topic description |
| `synthpersona generate-personas <questionnaire.json> [-n NUM] [-o FILE]` | Generate N personas (default 25) |
| `synthpersona simulate <questionnaire.json> <personas.json> [-o FILE]` | Simulate personas answering items |
| `synthpersona metrics <embeddings.json> -d DIM [-d DIM ...]` | Compute 6 diversity metrics |
| `synthpersona evolve [-i ITERS] [--islands NUM] [-q FILE ...]` | Run evolutionary optimization |

### Questionnaires

All 50 questionnaires from the paper (Appendix A.1) are included in `questionnaires/`, organized by split:

- **30 training** — used during evolution to evaluate generator fitness
- **10 validation** — used for early stopping during evolution
- **10 test** — held out for final evaluation

Three are hardcoded from the paper's Appendix A.2 (AGI Job Displacement, American Conspiracy Theories, Elderly Rural Japan). The remaining 47 were generated by our questionnaire generator following the paper's descriptions.

```bash
# Use any included questionnaire directly
uv run synthpersona generate-personas questionnaires/viking_warriors_valhalla.json

# Or generate a new one from scratch
uv run synthpersona generate-questionnaire "Attitudes toward space colonization in 2050"
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

The `evolve` command replicates the paper's AlphaEvolve optimization. It takes the baseline generator's Python code, uses an LLM to mutate it (one of 25 mutation strategies from the paper), evaluates the mutated version on diversity metrics, and keeps the best code per metric across parallel islands.

```bash
# Small test run
uv run synthpersona evolve -i 5 --islands 2

# Full run (paper settings: 500 iterations, 10 islands)
uv run synthpersona evolve -i 500 --islands 10 \
  -q questionnaires/climate.json \
  -q questionnaires/conspiracy.json
```

The best generator code is saved to `best_generator.py`.

## Configuration

All settings can be overridden via environment variables or `.env`:

| Setting | Default | Description |
|---------|---------|-------------|
| `VERTEXAI_PROJECT` | — | GCP project ID |
| `VERTEXAI_LOCATION` | `global` | Vertex AI region |
| `FAST_MODEL` | `vertex_ai/gemini-3-flash-preview` | Model for persona generation and simulation |
| `SMART_MODEL` | `vertex_ai/gemini-3-pro-preview` | Model for questionnaire generation and evolution |
| `POPULATION_SIZE` | `25` | Personas per population |
| `MAX_CONCURRENCY` | `10` | Max parallel LLM calls |
| `SIMULATION_TEMPERATURE` | `0.0` | Temperature for simulation (0 = deterministic) |

## Project Structure

```
synthpersona/
├── cli.py                        # Click CLI
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
│   └── two_stage.py              # Sobol sampling + parallel LLM expansion
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
