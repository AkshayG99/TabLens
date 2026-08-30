import torch
from trl import GRPOTrainer
from trl.trainer.utils import split_pixel_values_by_grid, split_tensor_dict, unsplit_pixel_values_by_grid

from disco_loss import compute_disco_policy_loss


class DiscoGRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        *args,
        disco_score_func: str = "logL",
        disco_delta: float = 1e-4,
        disco_beta: float = 1e3,
        disco_tau: float = 10.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.disco_score_func = disco_score_func
        self.disco_delta = disco_delta
        self.disco_beta = disco_beta
        self.disco_tau = disco_tau
        self._disco_reward_stash = None

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        rewards_per_func = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        self._disco_reward_stash = rewards_per_func.detach().clone()
        return rewards_per_func

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)

        if self._disco_reward_stash is None:
            raise RuntimeError(
                "DisCO: _calculate_rewards did not run before _generate_and_score_completions "
                "returned. TRL's internals may have changed since trl==0.24.0 in a way this "
                "override no longer matches -- do not silently continue with stale/None reward."
            )

        weights = self.reward_weights.to(self._disco_reward_stash.device)
        raw_rewards = (self._disco_reward_stash * weights.unsqueeze(0)).nansum(dim=1)
        self._disco_reward_stash = None  # one-shot; next call must re-stash

        binary_rewards = (raw_rewards >= 1.0).float()

        num_generations = self.num_generations
        batch_size = binary_rewards.shape[0]
        if batch_size % num_generations != 0:
            raise RuntimeError(
                f"DisCO: got {batch_size} completions, not a multiple of "
                f"num_generations={num_generations} -- can't form uniform prompt groups."
            )
        uid = torch.arange(batch_size, device=binary_rewards.device) // num_generations

        output["advantages"] = torch.stack([binary_rewards, uid.float()], dim=1)
        return output

    def _prepare_inputs(self, generation_batch):
        # Same as GRPOTrainer._prepare_inputs (trl 0.24.0) but WITHOUT the
        # shuffle_sequence_dict call. TRL shuffles the whole generation batch
        # before splitting it into per-accumulation-step micro-batches, which
        # scatters each prompt's num_generations completions across DIFFERENT
        # micro-batches. Standard GRPO survives that (advantages are already
        # group-normalized per row), but DisCO groups completions by uid inside
        # compute_loss -- it needs every micro-batch to contain the full
        # generation group, so the shuffle must be skipped.
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                generation_batch = self._generate_and_score_completions(generation_batch)
                generation_batch = split_pixel_values_by_grid(generation_batch)
                generation_batches = split_tensor_dict(
                    generation_batch, self.args.steps_per_generation
                )
                self._buffered_inputs = [
                    unsplit_pixel_values_by_grid(batch) for batch in generation_batches
                ]
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
            self._step += 1
        else:
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("DiscoGRPOTrainer does not support returning outputs")

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps, _entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=False,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        packed = inputs["advantages"]
        binary_rewards, uid = packed[:, 0], packed[:, 1].long()

        loss, metrics = compute_disco_policy_loss(
            old_per_token_logps=old_per_token_logps,
            per_token_logps=per_token_logps,
            binary_rewards=binary_rewards,
            uid=uid,
            completion_mask=completion_mask,
            num_generations=self.num_generations,
            score_func=self.disco_score_func,
            delta=self.disco_delta,
            beta=self.disco_beta,
            tau=self.disco_tau,
        )
        loss = loss / self.current_gradient_accumulation_steps

        mode = "train" if self.model.training else "eval"
        for key, value in metrics.items():
            self._metrics[mode][f"disco/{key}"].append(
                self.accelerator.gather(value.float().unsqueeze(0)).nanmean().item()
            )

        return loss
