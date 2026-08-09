# Copyright 2025 TabLens

# MIT License

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

import re
from typing import Optional


def _extract_verdict(text: str) -> Optional[str]:
    """Return 'accept' or 'reject' if the response contains a clear verdict."""
    if not text:
        return None

    # Explicit bold markers: **ACCEPT** / **REJECT**
    m = re.search(r"\*\*(ACCEPTED?|REJECTED?)\*\*", text, re.IGNORECASE)
    if m:
        return "accept" if m.group(1).lower().startswith("accept") else "reject"

    # "Decision: X" lines
    m = re.search(r"\bDecision\s*:?\s*\**\s*(ACCEPTED?|REJECTED?)\b", text, re.IGNORECASE)
    if m:
        return "accept" if m.group(1).lower().startswith("accept") else "reject"

    # Plain verdict words
    m = re.search(r"\b(ACCEPTED?|REJECTED?)\b", text, re.IGNORECASE)
    if m:
        return "accept" if m.group(1).lower().startswith("accept") else "reject"

    return None


def _target_to_verdict(target) -> str:
    """Map expected_response (0/1 or reject/accept strings) to a verdict."""
    if isinstance(target, str):
        t = target.strip().lower()
        if t in ("1", "good", "accept", "accepted"):
            return "accept"
        if t in ("0", "bad", "reject", "rejected"):
            return "reject"
        raise ValueError(f"Unrecognized expected_response string: {target!r}")
    return "accept" if int(target) == 1 else "reject"


def credit_reward_fn(data_source: str, solution_str: str, ground_truth, extra_info=None):
    """Binary credit-risk verdict reward.

    Returns 1.0 if the model's verdict matches expected_response, else 0.0.
    Expects the model output to contain ACCEPT/REJECT (e.g. "Decision: ACCEPT").
    """
    try:
        expected = _target_to_verdict(ground_truth)
    except (ValueError, TypeError):
        return 0.0

    predicted = _extract_verdict(solution_str)
    if predicted is None:
        return 0.0
    return 1.0 if predicted == expected else 0.0
