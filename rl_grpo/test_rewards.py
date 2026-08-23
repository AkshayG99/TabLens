"""Local self-check for rl_grpo/rewards.py (no torch/TRL needed).

Run:  python rl_grpo/test_rewards.py
"""

from rewards import (
    REWARD_CORRECT,
    REWARD_NO_VERDICT,
    REWARD_WRONG,
    credit_reward_fn,
    extract_verdict,
    score_completion,
    target_to_verdict,
)


def expect(name, got, want):
    assert abs(got - want) < 1e-9 if isinstance(want, float) else got == want, f"{name}: got {got!r}, want {want!r}"
    print(f"ok  {name}: {got!r}")


# --- verdict extraction -----------------------------------------------------
expect("bold", extract_verdict("The decision is **ACCEPTED** here"), "accept")
expect("decision-line", extract_verdict("blah\nDecision: REJECT"), "reject")
expect("final-decision", extract_verdict("Final decision: **REJECT**"), "reject")
expect("bare-word", extract_verdict("so we ACCEPT this application"), "accept")
expect("case-insensitive", extract_verdict("i would reject it"), "reject")
expect("none-when-absent", extract_verdict("the applicant seems okay overall"), None)
expect("none-empty", extract_verdict(""), None)

# --- label mapping ----------------------------------------------------------
expect("target-1", target_to_verdict(1), "accept")
expect("target-0", target_to_verdict(0), "reject")
expect("target-str-good", target_to_verdict("good"), "accept")

# --- dense scoring ----------------------------------------------------------
expect("correct-majority", score_completion("Final decision: ACCEPT", 1), REWARD_CORRECT)
expect("wrong-but-formatted", score_completion("Final decision: ACCEPT", 0), REWARD_WRONG)
expect("no-verdict", score_completion("nice profile, probably fine", 1), REWARD_NO_VERDICT)
# minority class (reject ground truth) is up-weighted
assert score_completion("Final decision: REJECT", 0) > REWARD_CORRECT, "minority weight not applied"
print(f"ok  minority-upweight: {score_completion('Final decision: REJECT', 0)} > {REWARD_CORRECT}")
# wrong on minority still just the small format credit
expect("wrong-minority", score_completion("Final decision: ACCEPT", 0), REWARD_WRONG)

# --- TRL entrypoint: chat-message completions -------------------------------
msgs = [[{"role": "assistant", "content": "Final decision: ACCEPT"}],
        [{"role": "assistant", "content": "Final decision: REJECT"}],
        [{"role": "assistant", "content": "cannot decide"}]]
out = credit_reward_fn(msgs, prompts=[None] * 3, target=[1, 0, 0])
expect("trl-msgs-correct", out[0], REWARD_CORRECT)
assert out[1] > REWARD_CORRECT, f"minority-correct not upweighted: {out[1]}"
print(f"ok  trl-msgs-minority-correct: {out[1]} > {REWARD_CORRECT}")
expect("trl-msgs-none", out[2], REWARD_NO_VERDICT)

# plain-string completions also work
out = credit_reward_fn(["Decision: REJECT"], target=["bad"])
assert out[0] > REWARD_CORRECT
print(f"ok  trl-strings: {out}")

print("\nALL PASS")
