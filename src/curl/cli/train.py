"""Train the pointer policy.

Settings come from configs/default.yaml. Flags override them.

Usage:  curl-train --updates 1200 --scenes 12 --rollouts 8
"""
from __future__ import annotations

from collections.abc import Sequence

from curl.cli.common import add_config_arg, make_parser
from curl.config import load_config, override
from curl.training.trainer import train


def build_parser():
    ap = make_parser(__doc__)
    add_config_arg(ap)
    ap.add_argument("--updates", type=int)
    ap.add_argument("--scenes", type=int)
    ap.add_argument("--rollouts", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--lr-min", type=float)
    ap.add_argument("--ent", type=float)
    ap.add_argument("--val-every", type=int)
    ap.add_argument("--val-scenes", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--ckpt", help="where to save the best checkpoint (overwrites v1 by default)")
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    cfg = override(cfg, "training", updates=args.updates, scenes=args.scenes,
                   rollouts=args.rollouts, lr=args.lr, lr_min=args.lr_min, ent=args.ent,
                   val_every=args.val_every, val_scenes=args.val_scenes, seed=args.seed)
    cfg = override(cfg, "paths", checkpoint=args.ckpt)
    train(cfg)


if __name__ == "__main__":
    main()
