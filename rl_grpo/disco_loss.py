import torch


# implementation from: https://github.com/verl-project/verl/pull/3357/changes#diff-af3da2c60785abde478f7bb68c303cd20e044e8af1b1ae93a2698f5b8fd5ed63
def compute_disco_policy_loss(
    old_per_token_logps: torch.Tensor,
    per_token_logps: torch.Tensor,
    binary_rewards: torch.Tensor,
    uid: torch.Tensor,
    completion_mask: torch.Tensor,
    num_generations: int,
    score_func: str = "logL",
    delta: float = 1e-4,
    beta: float = 1e3,
    tau: float = 10.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if num_generations < 2:
        raise ValueError(
            f"DisCO needs num_generations >= 2 to ever form a group with both a "
            f"correct and incorrect response (got {num_generations}). This exact "
            f"misconfiguration (rollout_n=1) is why the original verl DisCO attempt "
            f"trained with zero learning signal -- see GRPO_SETUP.md."
        )
    if not torch.all((binary_rewards == 0.0) | (binary_rewards == 1.0)):
        raise ValueError(
            "binary_rewards must be exactly 0.0 or 1.0 for every row -- binarize the "
            "shaped reward before calling compute_disco_policy_loss, don't pass it "
            "through directly."
        )

    negative_approx_kl = per_token_logps - old_per_token_logps
    ratio = torch.exp(negative_approx_kl)
    token_count = completion_mask.sum(dim=-1).clamp(min=1.0)
    ppo_kl = ((-negative_approx_kl) * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)

    if score_func == "logL":
        scores = (per_token_logps * completion_mask).sum(dim=-1) / token_count
    elif score_func == "Lratio":
        scores = (ratio * completion_mask).sum(dim=-1) / token_count
    else:
        raise ValueError(f"Unknown score_func {score_func!r}, expected 'logL' or 'Lratio'")

    sorted_uid, indices = uid.sort()
    sorted_scores = scores[indices]
    sorted_rewards = binary_rewards[indices]

    num_questions = uid.unique().numel()
    grouped_scores = sorted_scores.view(num_questions, num_generations)
    grouped_rewards = sorted_rewards.view(num_questions, num_generations)

    pos_mask = grouped_rewards == 1
    neg_mask = grouped_rewards == 0
    valid_mask = (pos_mask.sum(dim=1) != 0) & (neg_mask.sum(dim=1) != 0)

    if valid_mask.sum() > 0:
        grouped_scores = grouped_scores[valid_mask]
        pos_mask = pos_mask[valid_mask]
        neg_mask = neg_mask[valid_mask]

        neg_scores_masked = (grouped_scores / tau).masked_fill(~neg_mask, float("-inf"))
        neg_max, _ = neg_scores_masked.max(dim=-1, keepdim=True)
        neg_max = torch.where(neg_max == float("-inf"), torch.zeros_like(neg_max), neg_max)

        neg_exp = torch.exp(((grouped_scores / tau) - neg_max.detach()) * neg_mask) * neg_mask
        neg_sum_exp = neg_exp.sum(dim=-1, keepdim=True)
        neg_logmeanexp = neg_sum_exp / (neg_sum_exp.detach() + torch.finfo(neg_sum_exp.dtype).eps)

        pg_losses = ((grouped_scores - tau * neg_logmeanexp) * pos_mask).sum(dim=1, keepdim=True) / pos_mask.sum(
            dim=1, keepdim=True
        )
        pg_loss = pg_losses.sum() / num_questions
    else:
        pg_loss = torch.tensor(0.0, device=scores.device) * scores.mean()

    constraint = torch.maximum(beta * (ppo_kl - delta), torch.zeros_like(ppo_kl)).detach() * ppo_kl
    loss = -pg_loss + constraint

    metrics = {
        "ppo_kl": ppo_kl.detach(),
        "pg_clipfrac": (ppo_kl > delta).float().detach(),
        "valid_groups": valid_mask.sum().detach(),
        "total_groups": torch.tensor(float(num_questions), device=scores.device),
    }
    return loss, metrics
