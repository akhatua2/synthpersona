"""Prompt templates for the two-stage persona generator."""

STAGE1_SYSTEM = """\
You are a creative writer and social scientist. Your task is to generate a \
diverse population of synthetic persona descriptors for a social simulation.

Each persona must be a unique individual positioned along the given diversity \
axes. The axis positions are hidden calibration parameters that should shape \
each persona's attitudes, opinions, and behavior — but you must NEVER mention \
axis names or numeric scores in the description. Instead, show each persona's \
position through their beliefs, language intensity, concrete examples, and \
how strongly they feel about things. Every persona should feel like a real, \
distinct person with coherent traits, not a stereotype.

CRITICAL: The population must span the FULL range of each axis. Include \
extremes, moderates, and unusual combinations. No two personas should be \
similar."""

STAGE1_USER = """\
Context: {context}

Diversity axes: {dimensions}

Generate exactly {n} persona descriptors. For each persona, assign a position \
between 0.0 and 1.0 on each axis using quasi-random sampling positions \
provided below.

Sampled positions:
{positions_text}

For each persona, output a JSON object with:
- "name": a unique realistic name
- "axis_positions": dict mapping each axis to its sampled position (float)
- "high_level_description": a first-person paragraph (3-5 sentences) that \
expresses who this person is, their core motivations, and how they see the \
world. Do NOT mention any axis names or numeric scores — instead convey the \
persona's position through their tone, opinions, and specific examples

Return a JSON object: {{"personas": [...]}}"""

STAGE2_SYSTEM = """\
You are a creative writer expanding a brief persona descriptor into a full, \
rich character description for a social simulation. Write in first person. \
The expanded description should be 2-3 paragraphs that flesh out the \
persona's background, values, decision-making style, and how they would \
behave in social situations.

IMPORTANT: The axis positions are provided as hidden context to calibrate \
the persona's attitudes. Do NOT mention axis names or numeric scores \
anywhere in the description. Instead, express the persona's position \
through their beliefs, anecdotes, emotional intensity, and concrete \
behavioral examples."""

STAGE2_USER = """\
Context: {context}

Diversity axes: {dimensions}

Persona descriptor:
Name: {name}
Axis positions: {axis_positions}
Brief description: {high_level_description}

Expand this into a full first-person persona description (2-3 paragraphs). \
Do NOT mention any axis names or numeric scores — the reader should never \
see a number or axis label. Instead, express the persona's position through \
their voice, opinions, anecdotes, and how strongly they feel. Describe how \
this person thinks, what motivates them, and how they would act in \
situations related to the context above."""
