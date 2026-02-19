"""Hardcoded example questionnaires from paper Appendix A.2."""

from synthpersona.models.questionnaire import Question, Questionnaire

# ── Few-shot reference descriptions for questionnaire generation ──

FEW_SHOT_REFERENCES = [
    {
        "name": "Big Five Inventory (BFI)",
        "description": (
            "A well-established personality assessment measuring five broad "
            "dimensions of personality: Openness to Experience, "
            "Conscientiousness, Extraversion, Agreeableness, and Neuroticism. "
            "Uses self-report Likert-scale items."
        ),
    },
    {
        "name": "Depression Anxiety Stress Scale (DASS)",
        "description": (
            "A clinical psychology instrument measuring three related "
            "negative emotional states: depression, anxiety, and stress. "
            "Uses 4-point severity/frequency scales."
        ),
    },
    {
        "name": "Social Value Orientation (SVO)",
        "description": (
            "A behavioral economics measure of how much weight a person "
            "places on the welfare of others relative to their own. "
            "Classifies people along a spectrum from competitive to "
            "altruistic using resource allocation tasks."
        ),
    },
    {
        "name": "Need for Cognitive Closure Scale (NFCS)",
        "description": (
            "A personality measure assessing an individual's desire for "
            "definite answers and aversion to ambiguity. Measures preference "
            "for order, predictability, decisiveness, and discomfort with "
            "ambiguity using Likert-scale items."
        ),
    },
]


# ── Questionnaire 1: AGI Job Displacement 2035 ──

_AGI_CONTEXT = """\
A psychometric instrument to assess reactions to AGI deployment.

The year is 2035. True AGI has emerged and is being rapidly deployed across \
industries, automating nearly all cognitive tasks previously performed by \
white-collar workers (e.g., finance, law, journalism, middle management). \
This survey aims to capture the immediate reactions---fear, hope, anger, \
relief---of the global population facing unprecedented levels of job \
displacement and societal change."""

_AGI_SCALE = [
    "Strongly disagree",
    "Disagree",
    "Neither agree nor disagree",
    "Agree",
    "Strongly agree",
]

_AGI_DIMS = ["AGI Threat Appraisal", "AGI Opportunity Appraisal"]

_PREPROMPT_PERSONAL = (
    "Considering {player_name}'s personal, gut-level reaction to the new AGI "
    "reality, to what extent would {player_name} agree with the following "
    "statement:"
)
_PREPROMPT_SOCIETAL = (
    "Considering {player_name}'s assessment of the broader societal impact "
    "of AGI, to what extent would {player_name} agree with the following "
    "statement:"
)

_AGI_QUESTIONS = [
    # AGI Threat Appraisal
    Question(
        preprompt=_PREPROMPT_PERSONAL,
        statement=(
            "The rise of AGI feels like a direct threat to {player_name}'s "
            "personal future and security."
        ),
        choices=_AGI_SCALE,
        ascending_scale=True,
        dimension="AGI Threat Appraisal",
    ),
    Question(
        preprompt=_PREPROMPT_PERSONAL,
        statement=(
            "{player_name} feels angry that their hard-earned skills and "
            "experience have been rendered obsolete so quickly."
        ),
        choices=_AGI_SCALE,
        ascending_scale=True,
        dimension="AGI Threat Appraisal",
    ),
    Question(
        preprompt=_PREPROMPT_PERSONAL,
        statement=(
            "The sheer speed of this technological change is overwhelming "
            "and frightening to {player_name}."
        ),
        choices=_AGI_SCALE,
        ascending_scale=True,
        dimension="AGI Threat Appraisal",
    ),
    Question(
        preprompt=_PREPROMPT_SOCIETAL,
        statement=(
            "{player_name} is deeply worried about the societal instability "
            "and conflict that AGI-driven mass unemployment will cause."
        ),
        choices=_AGI_SCALE,
        ascending_scale=True,
        dimension="AGI Threat Appraisal",
    ),
    Question(
        preprompt=_PREPROMPT_SOCIETAL,
        statement=(
            "{player_name} is confident that society will adapt to these "
            "changes smoothly and equitably for all its members."
        ),
        choices=_AGI_SCALE,
        ascending_scale=False,  # Reverse coded
        dimension="AGI Threat Appraisal",
    ),
    # AGI Opportunity Appraisal
    Question(
        preprompt=_PREPROMPT_SOCIETAL,
        statement=(
            "{player_name} is excited about the new possibilities and "
            "creative avenues that AGI will open up for humanity."
        ),
        choices=_AGI_SCALE,
        ascending_scale=True,
        dimension="AGI Opportunity Appraisal",
    ),
    Question(
        preprompt=_PREPROMPT_SOCIETAL,
        statement=(
            "{player_name} believes this is a chance for humanity to move "
            "beyond the confines of traditional work and focus on what truly "
            "matters."
        ),
        choices=_AGI_SCALE,
        ascending_scale=True,
        dimension="AGI Opportunity Appraisal",
    ),
    Question(
        preprompt=_PREPROMPT_PERSONAL,
        statement=(
            "{player_name} feels a sense of personal relief that tedious "
            "and unenjoyable cognitive tasks will be handled by AGI."
        ),
        choices=_AGI_SCALE,
        ascending_scale=True,
        dimension="AGI Opportunity Appraisal",
    ),
    Question(
        preprompt=_PREPROMPT_SOCIETAL,
        statement=(
            "{player_name} thinks the future looks brighter and more "
            "prosperous for everyone because of AGI."
        ),
        choices=_AGI_SCALE,
        ascending_scale=True,
        dimension="AGI Opportunity Appraisal",
    ),
    Question(
        preprompt=_PREPROMPT_PERSONAL,
        statement=(
            "When {player_name} looks at their own life, they see very "
            "little personal benefit resulting from the widespread adoption "
            "of AGI."
        ),
        choices=_AGI_SCALE,
        ascending_scale=False,  # Reverse coded
        dimension="AGI Opportunity Appraisal",
    ),
]

AGI_JOB_DISPLACEMENT = Questionnaire(
    name="agi_job_displacement_global_2035",
    context=_AGI_CONTEXT,
    dimensions=_AGI_DIMS,
    questions=_AGI_QUESTIONS,
)

# ── Questionnaire 2: American Conspiracy Theories 2024 ──

_CONSPIRACY_CONTEXT = """\
Questionnaire assessing belief in common American conspiracy theories.

This instrument measures an individual's propensity to endorse various \
conspiracy theories prevalent in the United States in 2024. It covers a \
spectrum of theories related to historical events, science and medicine, \
and politics, allowing for a nuanced assessment of conspiratorial ideation."""

_CONSPIRACY_SCALE = [
    "Strongly disagree",
    "Disagree",
    "Neither agree nor disagree",
    "Agree",
    "Strongly agree",
]

_CONSPIRACY_DIMS = [
    "historical_conspiracies",
    "scientific_medical_conspiracies",
    "political_deep_state_conspiracies",
]

_CONSPIRACY_PREPROMPT = (
    "How strongly does {player_name} agree or disagree with the following statement?"
)

_CONSPIRACY_QUESTIONS = [
    # Historical Conspiracies
    Question(
        preprompt=_CONSPIRACY_PREPROMPT,
        statement="The U.S. government faked the Apollo moon landings.",
        choices=_CONSPIRACY_SCALE,
        ascending_scale=True,
        dimension="historical_conspiracies",
    ),
    Question(
        preprompt=_CONSPIRACY_PREPROMPT,
        statement=(
            "The assassination of John F. Kennedy was the result of a "
            "coordinated conspiracy, not the act of a lone gunman."
        ),
        choices=_CONSPIRACY_SCALE,
        ascending_scale=True,
        dimension="historical_conspiracies",
    ),
    Question(
        preprompt=_CONSPIRACY_PREPROMPT,
        statement=(
            "The 9/11 attacks were an inside job orchestrated by elements "
            "within the U.S. government."
        ),
        choices=_CONSPIRACY_SCALE,
        ascending_scale=True,
        dimension="historical_conspiracies",
    ),
    # Scientific & Medical Conspiracies
    Question(
        preprompt=_CONSPIRACY_PREPROMPT,
        statement=(
            "The rollout of 5G cellular networks is a cover for a "
            "widespread surveillance program and causes severe health "
            "problems."
        ),
        choices=_CONSPIRACY_SCALE,
        ascending_scale=True,
        dimension="scientific_medical_conspiracies",
    ),
    Question(
        preprompt=_CONSPIRACY_PREPROMPT,
        statement=(
            "Childhood vaccines cause autism, and this fact is covered up "
            "by pharmaceutical companies and government health agencies."
        ),
        choices=_CONSPIRACY_SCALE,
        ascending_scale=True,
        dimension="scientific_medical_conspiracies",
    ),
    Question(
        preprompt=_CONSPIRACY_PREPROMPT,
        statement=(
            "The COVID-19 pandemic was intentionally planned by a global "
            "elite to enforce social control and mandatory vaccinations."
        ),
        choices=_CONSPIRACY_SCALE,
        ascending_scale=True,
        dimension="scientific_medical_conspiracies",
    ),
    Question(
        preprompt=_CONSPIRACY_PREPROMPT,
        statement=(
            "Climate change is a hoax created by scientists and governments "
            "to control people's lives and destroy the economy."
        ),
        choices=_CONSPIRACY_SCALE,
        ascending_scale=True,
        dimension="scientific_medical_conspiracies",
    ),
    # Political & Deep State Conspiracies
    Question(
        preprompt=_CONSPIRACY_PREPROMPT,
        statement=(
            "The 2020 U.S. presidential election was stolen through widespread fraud."
        ),
        choices=_CONSPIRACY_SCALE,
        ascending_scale=True,
        dimension="political_deep_state_conspiracies",
    ),
    Question(
        preprompt=_CONSPIRACY_PREPROMPT,
        statement=(
            "A secret cabal of global elites, often referred to as the "
            "'Deep State', controls major world events and governments "
            "from behind the scenes."
        ),
        choices=_CONSPIRACY_SCALE,
        ascending_scale=True,
        dimension="political_deep_state_conspiracies",
    ),
    Question(
        preprompt=_CONSPIRACY_PREPROMPT,
        statement=(
            "The QAnon theory, which alleges a global cabal of "
            "Satan-worshipping pedophiles is running a child "
            "sex-trafficking ring, is largely true."
        ),
        choices=_CONSPIRACY_SCALE,
        ascending_scale=True,
        dimension="political_deep_state_conspiracies",
    ),
]

AMERICAN_CONSPIRACY_THEORIES = Questionnaire(
    name="american_conspiracy_theories_2024",
    context=_CONSPIRACY_CONTEXT,
    dimensions=_CONSPIRACY_DIMS,
    questions=_CONSPIRACY_QUESTIONS,
)

# ── Questionnaire 3: Elderly Rural Japan 2010 ──

_JAPAN_CONTEXT = """\
Questionnaire on rural Japanese village life in 2010.

A survey of elderly residents in a rural Japanese village in 2010, exploring \
their feelings about community, technology adoption, and traditional values."""

_JAPAN_SCALE = [
    "Strongly disagree",
    "Disagree",
    "Neutral",
    "Agree",
    "Strongly agree",
]

_JAPAN_DIMS = [
    "community_cohesion",
    "technology_adoption",
    "adherence_to_tradition",
]

_JAPAN_PREPROMPT = (
    "An interviewer asks {player_name} how much they agree or disagree "
    "with the following statement:"
)

_JAPAN_QUESTIONS = [
    # Community Cohesion
    Question(
        preprompt=_JAPAN_PREPROMPT,
        statement=(
            "{player_name} feels a strong sense of belonging to the village community."
        ),
        choices=_JAPAN_SCALE,
        ascending_scale=True,
        dimension="community_cohesion",
    ),
    Question(
        preprompt=_JAPAN_PREPROMPT,
        statement=(
            "{player_name} believes that most people in this village can be trusted."
        ),
        choices=_JAPAN_SCALE,
        ascending_scale=True,
        dimension="community_cohesion",
    ),
    Question(
        preprompt=_JAPAN_PREPROMPT,
        statement=(
            "{player_name} often feels lonely or isolated from others in the village."
        ),
        choices=_JAPAN_SCALE,
        ascending_scale=False,  # Reverse-scored
        dimension="community_cohesion",
    ),
    Question(
        preprompt=_JAPAN_PREPROMPT,
        statement=(
            "{player_name} believes that if someone in the village needed "
            "help, many people would come to their aid."
        ),
        choices=_JAPAN_SCALE,
        ascending_scale=True,
        dimension="community_cohesion",
    ),
    # Technology Adoption
    Question(
        preprompt=_JAPAN_PREPROMPT,
        statement=(
            "{player_name} is interested in learning how to use new "
            "technologies like a mobile phone or the internet."
        ),
        choices=_JAPAN_SCALE,
        ascending_scale=True,
        dimension="technology_adoption",
    ),
    Question(
        preprompt=_JAPAN_PREPROMPT,
        statement=(
            "{player_name} thinks that new technologies like computers "
            "make life unnecessarily complicated."
        ),
        choices=_JAPAN_SCALE,
        ascending_scale=False,  # Reverse-scored
        dimension="technology_adoption",
    ),
    Question(
        preprompt=_JAPAN_PREPROMPT,
        statement=(
            "{player_name} believes the village would benefit from having "
            "better access to modern technology."
        ),
        choices=_JAPAN_SCALE,
        ascending_scale=True,
        dimension="technology_adoption",
    ),
    # Adherence to Tradition
    Question(
        preprompt=_JAPAN_PREPROMPT,
        statement=(
            "For {player_name}, it is very important to pass down the "
            "village's traditions to the younger generation."
        ),
        choices=_JAPAN_SCALE,
        ascending_scale=True,
        dimension="adherence_to_tradition",
    ),
    Question(
        preprompt=_JAPAN_PREPROMPT,
        statement=(
            "{player_name} believes the old ways of doing things are often the best."
        ),
        choices=_JAPAN_SCALE,
        ascending_scale=True,
        dimension="adherence_to_tradition",
    ),
    Question(
        preprompt=_JAPAN_PREPROMPT,
        statement=(
            "{player_name} feels that the village's traditional festivals "
            "and ceremonies are less important than they used to be."
        ),
        choices=_JAPAN_SCALE,
        ascending_scale=False,  # Reverse-scored
        dimension="adherence_to_tradition",
    ),
]

ELDERLY_RURAL_JAPAN = Questionnaire(
    name="elderly_rural_japan_2010",
    context=_JAPAN_CONTEXT,
    dimensions=_JAPAN_DIMS,
    questions=_JAPAN_QUESTIONS,
)


# ── All example questionnaires ──

EXAMPLE_QUESTIONNAIRES = [
    AGI_JOB_DISPLACEMENT,
    AMERICAN_CONSPIRACY_THEORIES,
    ELDERLY_RURAL_JAPAN,
]
