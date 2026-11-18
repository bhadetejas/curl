"""Compare the CSV's own waypoint order with the RL policy's path.

Loads a waypoint CSV (x, y, z in mm, then unit normal nx, ny, nz), runs the saved policy
greedily to get the RL path, scores both orders on the same environment, and writes the
RL path (CSV and figure) and a metrics table. The CSV reference is its reachable
waypoints in file order.

Settings come from configs/default.yaml. Flags override them.

Usage:
    curl-plan
    curl-plan --csv path/to/points.csv --out results/my_run
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import torch

from curl.cli.common import add_config_arg, make_parser
from curl.config import load_config, load_run_settings, override, save_config
from curl.data.vanity import load_vanity
from curl.env.scoring import evaluate_order
from curl.evaluation.plotting import plot_path
from curl.evaluation.scene import build_scene_env
from curl.training.checkpoint import load_policy
from curl.training.rollout import rollout_batch

METRIC_COLUMNS = ["coverage_pct", "overlap_pct", "steps", "travel_m", "max_jump_m",
                  "off_surface_m", "joint_rad", "reward"]


def write_path_csv(positions, normals, order, path: Path) -> None:
    """Write the ordered path in mm, the same units as the input file."""
    rows = []
    for step, i in enumerate(order, start=1):
        p = positions[i] * 1000.0
        rows.append({"step": step, "waypoint": i, "x_mm": p[0], "y_mm": p[1], "z_mm": p[2],
                     "nx": normals[i][0], "ny": normals[i][1], "nz": normals[i][2]})
    pd.DataFrame(rows).to_csv(path, index=False)


def build_parser():
    ap = make_parser(__doc__)
    add_config_arg(ap)
    ap.add_argument("--csv", help="waypoint CSV (default: paths.input_csv)")
    ap.add_argument("--out", help="output prefix (default: <paths.results_dir>/vanity_rl)")
    ap.add_argument("--checkpoint", help="policy checkpoint (default: paths.checkpoint)")
    ap.add_argument("--base", type=float, nargs=3,
                    help="UR10e base position in metres (default: ik.base_pos_m)")
    ap.add_argument("--no-ik", action="store_true", help="skip the UR10e reachability check")
    ap.add_argument("--visit-all", action="store_true",
                    help="keep going until every reachable waypoint is visited")
    ap.add_argument("--allow-off-surface", action="store_true",
                    help="allow moves that leave the surface (e.g. across a sink cutout)")
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    run = load_run_settings(args.config)
    visit_all = args.visit_all or bool(run.get("visit_all", False))
    no_ik = args.no_ik or bool(run.get("no_ik", False))
    allow_off = args.allow_off_surface or bool(run.get("allow_off_surface", False))
    cfg = override(cfg, "paths", input_csv=args.csv, checkpoint=args.checkpoint)
    cfg = override(cfg, "ik", base_pos_m=args.base)
    out = Path(args.out) if args.out else cfg.paths.results_dir / "vanity_rl"

    positions, normals, _ = load_vanity(cfg.paths.input_csv)
    table_z = float(positions[:, 2].min())
    env = build_scene_env(positions, normals, cfg, table_z=table_z, use_ik=not no_ik,
                          full_visit=visit_all, forbid_off_surface=not allow_off)
    csv_order = [i for i in range(len(positions)) if env.reachable[i]]
    unreachable = [i for i in range(len(positions)) if not env.reachable[i]]
    print(f"{len(positions)} waypoints, {len(csv_order)} reachable, target cells T={env.T}")

    policy = load_policy(cfg.paths.checkpoint)
    with torch.no_grad():
        result = rollout_batch(policy, [env], torch.device("cpu"), greedy=True)
    rl_order = [int(i) for i in result.orders[0]]

    out.parent.mkdir(parents=True, exist_ok=True)
    write_path_csv(positions, normals, rl_order, Path(f"{out}_path.csv"))
    plot_path(positions, rl_order, f"{out}_coverage_path.png",
              title="RL pointer policy: coverage path", unreachable=unreachable)

    table = pd.DataFrame({"CSV file order": evaluate_order(env, csv_order),
                          "RL policy": evaluate_order(env, rl_order)}).T[METRIC_COLUMNS].round(2)
    table.to_csv(f"{out}_metrics.csv")
    save_config(cfg, Path(f"{out}_config.yaml"),
                run={"visit_all": visit_all, "no_ik": no_ik, "allow_off_surface": allow_off})
    with pd.option_context("display.width", 200):
        print(table.to_string())
    print(f"wrote {out}_path.csv, {out}_coverage_path.png, {out}_metrics.csv, {out}_config.yaml")


if __name__ == "__main__":
    main()
