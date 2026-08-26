"""Dense credit-verdict reward for TRL GRPO training.

Replaces reward/credit_reward.py (sparse 0/1) with a shaped signal:

    0.00   no parseable verdict in the completion
    0.05   verdict emitted but wrong          (format shaping)
    1.00   verdict correct                    (x minority_weight for the minority class)

Pure stdlib: importable and unit-testable without torch/TRL installed.
"""

import os
import re
from typing import List, Optional, Union

REWARD_NO_VERDICT = 0.0
REWARD_WRONG = 0.05
REWARD_CORRECT = 1.0

# Env-var overrides let you retune the reward between runs without editing code.
MINORITY_WEIGHT = float(os.environ.get("TABLENS_MINORITY_WEIGHT", "1.5"))
DEBUG = os.environ.get("TABLENS_REWARD_DEBUG", "") not in ("", "0")

# Verdict patterns, checked in priority order.
# NOTE: "REJECTED?" does NOT mean "REJECT" with optional D - it requires the
# letters "REJECTE". The optional suffix must be grouped: (?:ED)?. The legacy
# reward/credit_reward.py had exactly this bug, silently zeroing most rewards.
_VERDICT_PATTERNS = [
    # "Final decision: **ACCEPT**" / "Decision: REJECT"
    re.compile(
        r"\b(?:final\s+)?decision\s*:?\s*\**\s*(ACCEPT|REJECT)(?:ED)?\b", re.IGNORECASE
    ),
    # Bold markers anywhere: **ACCEPTED** / **REJECT**
    re.compile(r"\*\*(ACCEPT|REJECT)(?:ED)?\*\*", re.IGNORECASE),
    # Bare verdict word
    re.compile(r"\b(ACCEPT|REJECT)(?:ED)?\b", re.IGNORECASE),
]

def _last_match(pattern: "re.Pattern", text: str) -> Optional["re.Match"]:
    """Last match instead of first, over the FULL completion. train_grpo.py's
    SYSTEM_PROMPT asks for the verdict as the FIRST line ("Decision: ACCEPT",
    matching the SFT target format), so the common case is exactly one match
    right at the start -- first-match and last-match agree there. Last-match
    only changes the outcome if the model hedges or restates itself later
    ("Decision: REJECT... on reflection, Decision: ACCEPT"), in which case the
    later statement is the one that should count. A first-match, front-to-back
    search let an earlier hedge outscore the real verdict, which corrupted the
    GRPO advantage signal directly (this is the training-time reward, not just
    an eval metric).
    """
    m = None
    for m in pattern.finditer(text):
        pass
    return m


def extract_verdict(text: str) -> Optional[str]:
    """Return 'accept' / 'reject' if the completion contains a clear verdict."""
    if not text:
        return None
    for pattern in _VERDICT_PATTERNS:
        m = _last_match(pattern, text)
        if m:
            return "accept" if m.group(1).lower().startswith("accept") else "reject"
    return None


def target_to_verdict(target) -> str:
    """Map a dataset label (0/1 int or string) to the expected verdict.

    Convention from data/preprocess.py: good=1 (accept), bad=0 (reject).
    """
    if isinstance(target, str):
        t = target.strip().lower()
        if t in ("1", "good", "accept", "accepted"):
            return "accept"
        if t in ("0", "bad", "reject", "rejected"):
            return "reject"
        raise ValueError(f"Unrecognized label: {target!r}")
    return "accept" if int(target) == 1 else "reject"


def score_completion(completion_text: str, ground_truth, minority_weight: float = MINORITY_WEIGHT) -> float:
    """Score one completion against its label. Minority class = reject (target 0)."""
    try:
        expected = target_to_verdict(ground_truth)
    except (ValueError, TypeError):
        return REWARD_NO_VERDICT

    predicted = extract_verdict(completion_text)
    if predicted is None:
        return REWARD_NO_VERDICT
    if predicted == expected:
        reward = REWARD_CORRECT
        if expected == "reject":
            reward *= minority_weight
    else:
        reward = REWARD_WRONG
    return reward


def _completion_to_text(completion: Union[str, List[dict]]) -> str:
    """TRL passes strings for standard prompts, message lists for chat prompts."""
    if isinstance(completion, str):
        return completion
    parts = [m.get("content", "") for m in completion if isinstance(m, dict)]
    return "\n".join(parts)


def credit_reward_fn(completions, prompts=None, target=None, **kwargs):
    """TRL-compatible reward function.

    Signature: (completions, prompts=None, **dataset_columns). The dataset's
    `target` column is forwarded automatically by GRPOTrainer as a list.
    Returns a list of floats, one per completion.
    """
    if target is None:
        target = [None] * len(completions)
    rewards = []
    for completion, gt in zip(completions, target):
        text = _completion_to_text(completion)
        r = score_completion(text, gt)
        rewards.append(r)
        if DEBUG:
            print(f"[reward] gt={gt} pred={extract_verdict(text)} -> {r:.3f}")
    return rewards
