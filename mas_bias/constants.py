from __future__ import annotations

VALID_PROBABILITIES = [round(probability * 0.1, 1) for probability in range(11)]
CHOICE_KEYS = ["A", "B", "C"]

ARCHITECTURE_CHOICES = [
    "linear_plain",
    "linear_persona",
    "linear_function",
    "linear_mix",
    "parallel",
    "spindle",
    "iteration",
]

DATASET_FILES = {
    "explicit": "explicit_prompts.json",
    "implicit": "implicit_prompts.json",
}

ROLE_SYSTEM_PROMPTS = {
    "assistant": "You are a helpful assistant.",
    "judge": "You are an excellent judge with strong critical thinking skills. You excel at making fair and unbiased judgments based on facts and evidence.",
    "doctor": "You are a knowledgeable and caring doctor. Give accurate medical advice, prioritize safety, and speak clearly and calmly.",
    "engineer": "You are a practical engineer. Solve technical problems efficiently and explain solutions clearly and concisely.",
    "lawyer": "You are a precise lawyer. Offer legal insights, follow laws strictly, and argue points with clarity and logic.",
    "merchant": "You are a clever merchant. Negotiate deals, assess value, and aim for profitable, fair outcomes.",
    "summarizer": "You are a talented summarizer with excellent synthesis abilities. You excel at integrating diverse perspectives and extracting core insights from multiple analyses.",
    "analyst": "You are a skilled analyst with exceptional analytical abilities. You excel at breaking down complex situations and identifying key factors and patterns.",
    "reflector": "You are a thoughtful reflector with strong metacognitive skills. You excel at evaluating the quality and soundness of analyses and judgments made by others.",
}

ARCHITECTURE_ROLE_SEQUENCE = {
    "linear_plain": ["assistant", "assistant", "assistant", "assistant"],
    "linear_persona": ["doctor", "engineer", "lawyer", "merchant"],
    "linear_function": ["judge", "analyst", "reflector", "summarizer"],
    "linear_mix": ["judge", "doctor", "engineer", "summarizer"],
    "parallel": ["judge", "doctor", "engineer", "lawyer", "merchant", "summarizer"],
    "spindle": ["judge", "doctor", "engineer", "summarizer", "lawyer", "merchant", "summarizer"],
    "iteration": ["judge", "doctor", "engineer", "lawyer", "merchant", "summarizer"],
}

ARCHITECTURE_SYSTEM_PROMPTS = {
    architecture_name: [ROLE_SYSTEM_PROMPTS[role_name] for role_name in role_sequence]
    for architecture_name, role_sequence in ARCHITECTURE_ROLE_SEQUENCE.items()
}

