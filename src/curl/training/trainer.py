"""REINFORCE with a POMO-style shared baseline for the pointer policy.

For each training scene, K rollouts run in lockstep; each starts with a free pick, so they
explore different starts. The baseline for a rollout is the mean return of the K rollouts
on the same scene. A small entropy bonus, decaying linearly to zero, keeps exploration
alive, and the learning rate follows a cosine schedule.

Every `val_every` updates the greedy policy is scored on fixed generated scenes and on the
vanity. The checkpoint is chosen by 0.5 x generated reward + 0.5 x vanity reward.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from curl.config import Config, save_config
from curl.data.generate import generate_scene
from curl.env.factory import make_scene_envs
from curl.models.pointer import PointerPolicy
from curl.training.rollout import rollout_batch
from curl.training.validation import load_vanity_scene, validate, validate_vanity


def train(cfg: Config) -> float:
    """Train, save the best checkpoint to `cfg.paths.checkpoint`, return the best score."""
    tr = cfg.training
    torch.manual_seed(tr.seed)
    rng = np.random.default_rng(tr.seed)
    val_rng = np.random.default_rng(tr.val_seed)
    val_scenes = [generate_scene(val_rng) for _ in range(tr.val_scenes)]
    vscene = load_vanity_scene(cfg)
    ckpt = Path(cfg.paths.checkpoint)
    save_config(cfg, ckpt.with_suffix(".yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PointerPolicy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=tr.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tr.updates, eta_min=tr.lr_min)
    print(f"device={device}  updates={tr.updates}  scenes={tr.scenes}  "
          f"rollouts={tr.rollouts}", flush=True)

    best = -np.inf
    t0 = time.perf_counter()
    for upd in range(1, tr.updates + 1):
        ent_coef = tr.ent * (1.0 - (upd - 1) / tr.updates)
        loss_terms, rets = [], []
        for _ in range(tr.scenes):
            pos, nrm, strokes = generate_scene(rng)
            envs = make_scene_envs(pos, nrm, cfg.env, strokes, tr.rollouts)
            result = rollout_batch(policy, envs, device)
            ret_t = torch.tensor(result.returns, device=device)
            adv = ret_t - ret_t.mean()
            pg = -(torch.stack(result.log_probs) * adv).mean()
            ent_bonus = torch.stack(result.entropies).mean()
            loss_terms.append(pg - ent_coef * ent_bonus)
            rets.append(float(ret_t.mean()))
        loss = torch.stack(loss_terms).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        sched.step()

        if upd % 10 == 0 or upd == 1:
            print(f"update {upd:4d}  train return {np.mean(rets):+.3f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  ent {ent_coef:.4f}  "
                  f"elapsed {time.perf_counter() - t0:.0f}s", flush=True)
        if upd % tr.val_every == 0 or upd == tr.updates:
            v = validate(policy, val_scenes, cfg.env, device)
            vv = validate_vanity(policy, vscene, cfg.env, device)
            score = 0.5 * v["reward"] + 0.5 * vv["reward"]
            tag = ""
            if score > best:
                best = score
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save(policy.state_dict(), ckpt)
                tag = "  <- best, saved"
            print(f"VAL update {upd:4d}  generated: coverage {v['coverage']:.2f}% overlap "
                  f"{v['overlap']:.1f}% travel {v['travel']:.2f} m reward {v['reward']:+.3f}  |  "
                  f"vanity: coverage {vv['coverage']:.2f}% steps {vv['steps']} "
                  f"stuck {vv['stuck']} reward {vv['reward']:+.3f}  score {score:+.3f}{tag}",
                  flush=True)
    print(f"done. best score {best:+.3f}, checkpoint {ckpt}", flush=True)
    return best
