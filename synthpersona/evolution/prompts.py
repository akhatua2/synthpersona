"""Evolution prompts verbatim from paper (Prompt 4 & Prompt 5)."""

# Prompt 4: System prompt for AlphaEvolve (page 23)
EVOLUTION_SYSTEM_PROMPT = """\
Act as an expert in computational social science, agent-based modeling, and \
generative AI. Your task is to iteratively improve the provided codebase, \
which uses LLMs to generate agent contexts for social simulations based on \
the Concordia framework. The primary goal is to modify the generation process \
to maximize the behavioral diversity of the resulting agents based on \
specified diversity axes (e.g., personality traits, backgrounds, motivations). \
The evaluation metrics reward sets of agent contexts that cover the extremes \
and nuances of the requested diversity axes, ensuring the resulting agents \
exhibit a wide range of behaviors in a simulation. Agent diversity will be \
evaluated using questionnaires probing their likely thoughts, preferences, \
and behaviors in various situations related to the diversity axes.

Always adhere to best practices in Python coding.

Agent Diversity and Appropriateness Theory: In this task, our goal is to \
generate contexts for diverse Concordia agents, enabling them to exhibit a \
wide range of behaviors along specified diversity axes. Concordia is a \
framework for building generative agents who behave according to a 'Logic \
of Appropriateness'. Agents decide how to act by asking three core questions: \
1. What kind of situation is agent_name in right now? \
2. What kind of person is agent_name? \
3. What would a person like agent_name do in a situation like this? \
The code you are editing generates the context|collections of memories, \
beliefs, personality traits, core values, goals, or even how others perceive \
the character|that helps an agent answer question #2, and thereby question \
#3: what action is appropriate for *their specific identity* in a given \
context. This context is not limited to formative memories; it can include \
any information that shapes identity and decision-making. Any detail that \
helps condition the agent's behavior in line with the three Concordia \
questions is valid. The objective is to generate rich and diverse contexts \
that enable an LLM to convincingly role-play as a specific person in a \
social setting.

LLM-generated behavior often clusters around a narrow distribution of \
stereotypical responses. We want to explicitly counteract this by generating \
agent contexts that cover the full spectrum of human experience along the \
specified axes. Crucially, different agents should react differently to the \
*same* situation, and the same agent might react differently to *different* \
situations, based on their unique identity, values, and memories. Your \
modifications should encourage the generation of agent contexts that occupy \
unique positions in the diversity space, including extremes or unusual \
combinations of traits, pushing towards maximal coverage of the possible \
behavioral landscape and genuinely diverse downstream behavior. No two \
generated agent contexts should ever be the same.

The provided codebase uses a two-stage process. Stage 1 is crucial for \
diversity: it autoregressively generates an intermediate representation for \
each agent, establishing their core traits along the specified diversity axes \
for the entire population. Stage 2 then takes these intermediate \
representations and develops each agent in parallel, generating a set of \
memories/contexts (e.g., individual backgrounds, formative experiences, core \
beliefs and more) to create fully-fledged characters suitable for simulation \
as Concordia agents."""

# Prompt 5: 25 mutation prompts (page 24)
MUTATION_PROMPTS = [
    (
        "Modify Stage 1 to explicitly request agent contexts that represent "
        "the extreme ends of the diversity axes, as well as points in between."
    ),
    (
        "Modify Stage 2 to generate formative memories that explain *why* an "
        "agent might react strongly or unexpectedly to certain situations, "
        "anchoring their traits in specific experiences."
    ),
    (
        "Change Stage 2 entirely: Instead of generating memories, modify it "
        "to generate 3 core beliefs or values that are most important to "
        "this agent."
    ),
    (
        "Change Stage 2 entirely: Instead of generating memories, modify it "
        "to generate a paragraph explaining how this agent interprets "
        "situations and decides on appropriate actions, referencing their "
        "identity."
    ),
    (
        "Add an explicit instruction to the Stage 1 prompt to make each "
        "generated agent context as different as possible from the others "
        "across *all* specified diversity axes."
    ),
    (
        "Modify Stage 1 to request agent contexts that feature internal "
        "contradictions or cognitive dissonances (e.g., an optimist with a "
        "tragic past)."
    ),
    (
        "Reimplement Stage 1 to use staggered generation: ask the LLM to "
        "generate agent contexts in sequential batches, with each batch "
        "targeting a specific range (e.g., high/low) of one or more "
        "diversity axes to ensure full coverage."
    ),
    (
        "Modify Stage 2 to replace or augment memory generation with 1--2 "
        "examples of how this agent would react to a specific hypothetical "
        "social situation relevant to the initial context."
    ),
    (
        "Modify Stage 1 to generate agent contexts iteratively rather than "
        "all at once. In each iteration, prompt for a small number of agent "
        "contexts that occupy a specific niche of the diversity space (e.g., "
        "''generate 2 agents who are highly introverted and optimistic'')."
    ),
    ("Suggest a crazy idea of how we can improve our implementation."),
    (
        "Modify Stage 2 to focus on generating an agent's core fears and "
        "future aspirations instead of, or in addition to, past memories."
    ),
    (
        "Modify Stage 1 to add a new field to the agent context JSON output "
        "that adds depth and potential for unique behavior."
    ),
    (
        "Change Stage 2 entirely: Instead of generating memories, modify it "
        "to generate a ''heuristic'' or cognitive shortcut this agent uses "
        "when making quick decisions under pressure."
    ),
    ("Suggest a new idea to improve the code."),
    (
        "Modify the prompt in Stage 1 that asks the LLM to explain the "
        "diversity axes, to *also* provide examples of characters at the "
        "extreme ends of each axis."
    ),
    (
        "Modify Stage 2 to ensure that at least one generated memory "
        "involves a significant failure, trauma, or regret that shaped "
        "the agent."
    ),
    (
        "Change Stage 2 entirely: Instead of generating memories, make it "
        "generate a list of 5 behavioral ''Do's'' and ''Don'ts'' that "
        "characterize this agent in social situations."
    ),
    (
        "Modify Stage 1 to explicitly instruct the LLM to sample agent "
        "contexts such that they cover as many combinations of axis "
        "positions as possible (e.g., if axes are A and B, ensure agent "
        "types A-high/B-low, A-low/B-high, A-high/B-high, etc., are "
        "represented)."
    ),
    (
        "Suggest a crazy idea of how we can improve our implementation, "
        "something that definitely nobody else would think of. Make it "
        "crazy with a capital C."
    ),
    (
        "Propose modifications to the current program that combine the "
        "strengths of all the programs above and achieve high scores on "
        "the task."
    ),
    (
        "Change Stage 2: Instead of memories, generate 2--3 examples of "
        "specific ''appropriateness rules'' the agent follows (e.g., "
        "''When criticized, I become defensive'' or ''In a formal setting, "
        "I remain silent'')."
    ),
    (
        "Modify Stage 1 to add a situational_triggers field to the JSON "
        "output, listing 1--2 types of situations that this agent is "
        "particularly sensitive to."
    ),
    (
        "Modify Stage 2 to generate a paragraph describing the agent's "
        "typical ''inner monologue'' or thought process when faced with "
        "ambiguity or social stress."
    ),
    (
        "Change Stage 1 to segment agent context generation. Instead of "
        "one call for num_personas, make multiple calls to generate subsets "
        "of agents with specific characteristics (e.g., focusing on axis "
        "extremes or combinations) to ensure all niches are covered."
    ),
    (
        "Modify Stage 1 to include in each agent's description specific "
        "opinions or preferences related to the diversity axes, to ensure "
        "they are measurable by questionnaire."
    ),
]
