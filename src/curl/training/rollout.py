"""Lockstep rollouts of the pointer policy over one or more environments."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from curl.models.pointer import PointerPolicy, node_features


@dataclass
class RolloutResult:
    """Per-environment outputs of `rollout_batch`, in the order the envs were given."""

    log_probs: list[torch.Tensor]
    returns: list[float]
    steps: list[int]
    orders: list[list[int]]
    entropies: list[torch.Tensor]


def rollout_batch(policy: PointerPolicy, envs, device, greedy: bool = False) -> RolloutResult:
    """Run several envs in lockstep, sampling from the policy unless `greedy`.

    Each env is reset first and left in its final state, so its `stats` can be read after.
    Entropy is summed over selectable entries only, since -inf * 0 would give NaN gradients.
    """
    K = len(envs)
    obs, infos = [], []
    for env in envs:
        o, info = env.reset()
        obs.append(o)
        infos.append(info)
    centre = obs[0]["positions"][:, :2].mean(0)
    scale = max(float(np.ptp(obs[0]["positions"][:, :2], axis=0).max()), 1e-6)
    logps = [[] for _ in range(K)]
    ents = [[] for _ in range(K)]
    orders = [[] for _ in range(K)]
    done = [False] * K
    ret = [0.0] * K
    steps = [0] * K

    while not all(done):
        feats = torch.tensor(np.stack([node_features(o, scale, centre) for o in obs]),
                             device=device)
        cur = torch.tensor([o["current"] for o in obs], device=device)
        mask = torch.tensor(np.stack([inf["action_mask"] for inf in infos]), device=device)
        mask_any = mask.any(dim=1)
        sel = mask.clone().masked_fill(~mask_any.unsqueeze(1), True)
        logp = policy(feats, cur, sel)
        probs = logp.exp()
        if greedy:
            actions = probs.argmax(dim=1)
        else:
            actions = torch.multinomial(probs.detach(), 1).squeeze(1)
        logp_s = torch.where(sel, logp, torch.zeros_like(logp))
        probs_s = torch.where(sel, probs, torch.zeros_like(probs))
        step_ent = -(probs_s * logp_s).sum(dim=1)
        for k in range(K):
            if done[k] or not mask_any[k]:
                done[k] = True
                continue
            a = int(actions[k])
            logps[k].append(logp[k, a])
            ents[k].append(step_ent[k])
            orders[k].append(a)
            o, r, term, trunc, info = envs[k].step(a)
            obs[k], infos[k] = o, info
            ret[k] += r
            steps[k] += 1
            if term or trunc:
                done[k] = True

    zero = torch.zeros((), device=device)
    return RolloutResult(
        log_probs=[torch.stack(lp).sum() if lp else zero for lp in logps],
        returns=ret,
        steps=steps,
        orders=orders,
        entropies=[torch.stack(e).sum() if e else zero for e in ents],
    )
