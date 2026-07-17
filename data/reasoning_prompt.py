SYSTEM_PROMPT = (
    "You are a credit risk analyst. Given a loan applicant's profile, "
    "analyze each feature to assess creditworthiness. Think step-by-step, "
    "then provide your final classification.\n\n"
    "You MUST output exactly in this format:\n"
    "<reasoning>[Your step-by-step analysis]</reasoning>\n"
    "<answer>[good or bad]</answer>"
)

GERMAN_REASONING_INSTRUCTIONS = (
    "Analyze the following German Credit applicant profile.\n"
    "Consider:\n"
    "- Age and job stability (higher job level = more stable)\n"
    "- Housing situation (own > rent > free)\n"
    "- Savings and checking account levels\n"
    "- Credit amount relative to duration\n"
    "- Loan purpose risk\n\n"
    "Profile: {text}\n\n"
    "Provide step-by-step reasoning, then classify as 'good' or 'bad'."
)

LOAN_REASONING_INSTRUCTIONS = (
    "Analyze the following LendingClub loan applicant profile.\n"
    "Consider:\n"
    "- Employment stability and income level\n"
    "- Debt-to-income ratio (lower is better)\n"
    "- Credit history: delinquencies, credit lines, utilization\n"
    "- Loan grade and interest rate\n"
    "- Account delinquency percentage\n\n"
    "Profile: {text}\n\n"
    "Provide step-by-step reasoning, then classify as 'good' or 'bad'."
)

DATASET_INSTRUCTIONS = {
    "german": GERMAN_REASONING_INSTRUCTIONS,
    "loans": LOAN_REASONING_INSTRUCTIONS,
}

MODEL_NAME = "gemma-4-31b-it"


def build_prompt(text: str, dataset: str):
    from google.genai import types
    instruction = DATASET_INSTRUCTIONS[dataset].format(text=text)
    return [
        types.Content(role="user", parts=[types.Part.from_text(text=f"[System] {SYSTEM_PROMPT}\n\n{instruction}")]),
    ]


def parse_response(response: str) -> tuple[str, str] | None:
    import re
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", response, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if not reasoning_match or not answer_match:
        return None
    reasoning = reasoning_match.group(1).strip()
    answer = answer_match.group(1).strip().lower()
    if answer not in ("good", "bad"):
        return None
    return reasoning, answer
