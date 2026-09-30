"""Token-level PPO objectives for sampled assistant actions (never user tokens)."""

from __future__ import annotations

import torch


def action_statistics(model, critic, input_ids, prompt_length):
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or not 0 < prompt_length < input_ids.shape[1]:
        raise ValueError("expected one complete prompt+assistant sequence")
    output = model(
        input_ids=input_ids, attention_mask=torch.ones_like(input_ids), output_hidden_states=True, use_cache=False
    )
    logits = output.logits[0, prompt_length - 1 : -1].float()
    actions = input_ids[0, prompt_length:]
    log_distribution = logits.log_softmax(-1)
    log_probs = log_distribution.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    entropy = -(log_distribution.exp() * log_distribution).sum(-1)
    # A value head shares features with the policy; critic loss must not update
    # the actor backbone. Values refer to states BEFORE each assistant action.
    hidden = output.hidden_states[-1][0, prompt_length - 1 : -1].detach().float()
    values = critic(hidden).squeeze(-1)
    return log_probs, values, entropy


def sampled_kl(log_probs, reference_log_probs):
    difference = reference_log_probs - log_probs
    result = torch.expm1(difference) - difference
    if not torch.isfinite(result).all():
        raise ValueError("nonfinite reference KL")
    return result.clamp_min(0)


def generalized_advantages(rewards, values, *, gamma=1.0, lam=0.95):
    """A complete episode ends after the final assistant action; terminal V=0.

    Concatenating assistant segments carries future-turn credit across scripted
    environment/user turns, without optimizing those environment tokens.
    """
    if rewards.ndim != 1 or rewards.shape != values.shape or not rewards.numel():
        raise ValueError("episode rewards and values must be equal nonempty vectors")
    if not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError("gamma and lambda must be in [0,1]")
    advantages = torch.zeros_like(rewards)
    carry, next_value = torch.zeros_like(rewards[0]), torch.zeros_like(values[0])
    for index in range(rewards.numel() - 1, -1, -1):
        delta = rewards[index] + gamma * next_value - values[index]
        carry = delta + gamma * lam * carry
        advantages[index] = carry
        next_value = values[index]
    return advantages.detach(), (advantages + values).detach()


def clipped_loss(
    log_probs,
    values,
    entropy,
    old_log_probs,
    old_values,
    advantages,
    returns,
    *,
    clip_range=0.2,
    value_clip=0.2,
    value_coef=0.5,
    entropy_coef=0.0,
):
    tensors = (log_probs, values, entropy, old_log_probs, old_values, advantages, returns)
    if any(value.shape != log_probs.shape for value in tensors) or log_probs.ndim != 1:
        raise ValueError("PPO tensors must have matching token dimensions")
    if any(not torch.isfinite(value).all() for value in tensors):
        raise ValueError("nonfinite PPO inputs")
    log_ratio = log_probs - old_log_probs.detach()
    ratio = log_ratio.exp()
    advantage = advantages.detach()
    policy_loss = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(1 - clip_range, 1 + clip_range)).mean()
    clipped_values = old_values.detach() + (values - old_values.detach()).clamp(-value_clip, value_clip)
    value_loss = (
        0.5 * torch.maximum((values - returns.detach()).square(), (clipped_values - returns.detach()).square()).mean()
    )
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()
    if not torch.isfinite(loss):
        raise ValueError("nonfinite PPO loss")
    metrics = {
        "policy_loss": policy_loss.detach().item(),
        "value_loss": value_loss.detach().item(),
        "entropy": entropy.detach().mean().item(),
        "approx_policy_kl": sampled_kl(old_log_probs.detach(), log_probs.detach()).mean().item(),
        "clip_fraction": ((ratio - 1).abs() > clip_range).float().mean().detach().item(),
    }
    return loss, metrics
