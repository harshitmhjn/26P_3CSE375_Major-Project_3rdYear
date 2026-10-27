"""Train DRL scheduling agents (PPO and/or DQN) across several training seeds.

    python scripts/train_agent.py --algo PPO --timesteps 300000 --seeds 0 1 2
    python scripts/train_agent.py --algo DQN --timesteps 200000 --seeds 0 1 2
    python scripts/train_agent.py --algo PPO --timesteps 5000 --seeds 0 --outdir results/smoke

Training is resumable: re-running the same command continues any interrupted
run from its last checkpoint (written every --checkpoint-every timesteps).
Training seeds must not overlap the held-out evaluation seeds (1001-1005).
Each completed run is evaluated on the held-out seeds and a JSON record is
written next to the checkpoint.
"""

import argparse
import json

import _bootstrap  # noqa: F401

from dcsched import EnvConfig
from dcsched.training import train_run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", default="PPO", choices=["PPO", "DQN", "ppo", "dqn"])
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--regime", default="steady", choices=["steady", "diurnal", "bursty"])
    ap.add_argument("--episode-length", type=int, default=1000)
    ap.add_argument("--config", default=None, help="optional EnvConfig JSON")
    ap.add_argument("--alpha", type=float, default=None, help="thermal reward weight")
    ap.add_argument("--beta", type=float, default=None, help="energy reward weight")
    ap.add_argument("--gamma-w", type=float, default=None, help="performance reward weight")
    ap.add_argument("--outdir", default="results/models")
    ap.add_argument("--checkpoint-every", type=int, default=25_000)
    ap.add_argument("--no-resume", action="store_true", help="ignore existing checkpoints")
    args = ap.parse_args()

    algo = args.algo.upper()
    cfg = EnvConfig.from_json(args.config) if args.config else EnvConfig()
    cfg.workload.regime = args.regime
    cfg.episode_length = args.episode_length
    if args.alpha is not None:
        cfg.reward.alpha = args.alpha
    if args.beta is not None:
        cfg.reward.beta = args.beta
    if args.gamma_w is not None:
        cfg.reward.gamma_w = args.gamma_w

    print(f"Training {algo} | {args.timesteps:,} timesteps | seeds {args.seeds} | "
          f"regime {args.regime} | weights a={cfg.reward.alpha} b={cfg.reward.beta} g={cfg.reward.gamma_w}")
    for seed in args.seeds:
        rec = train_run(algo, cfg, seed, args.timesteps, args.outdir,
                        resume=not args.no_resume, checkpoint_every=args.checkpoint_every)
        ev = rec.get("eval", {})
        msg = (f"seed {seed}: {rec['status']} at {rec['timesteps_done']:,} steps "
               f"in {rec['train_seconds']:.0f}s")
        if ev:
            msg += (f" | held-out hotspot {ev['hotspot_frequency_mean']:.3f}, "
                    f"completed {ev['tasks_completed_mean']:.1f}, return {ev['episode_return_mean']:.0f}")
        print(msg)

    with open(f"{args.outdir}/{algo.lower()}_config.json", "w") as fh:
        json.dump(cfg.to_dict(), fh, indent=2)


if __name__ == "__main__":
    main()
