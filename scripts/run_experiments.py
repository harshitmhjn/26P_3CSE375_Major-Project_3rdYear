"""Run the evaluation pipeline: baselines + every trained DRL checkpoint found.

    python scripts/run_experiments.py                       # steady regime
    python scripts/run_experiments.py --regimes steady diurnal bursty
    python scripts/run_experiments.py --no-drl              # baselines only
    python scripts/run_experiments.py --config config/default.json --name myrun

Writes results/experiments/<name>/ (raw CSV, summary CSV, config, report.md,
figures). Evaluation always uses the held-out seeds in dcsched.evaluate.
"""

import argparse

import _bootstrap  # noqa: F401  (sets sys.path / cwd)

from dcsched import EnvConfig
from dcsched.experiments import run_experiment


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regimes", nargs="+", default=["steady"],
                    choices=["steady", "diurnal", "bursty"])
    ap.add_argument("--config", default=None, help="optional EnvConfig JSON to start from")
    ap.add_argument("--episode-length", type=int, default=None)
    ap.add_argument("--name", default=None, help="experiment name (default: <regime>)")
    ap.add_argument("--models-dir", default="results/models")
    ap.add_argument("--out", default="results/experiments")
    ap.add_argument("--no-drl", action="store_true", help="evaluate baselines only")
    args = ap.parse_args()

    for regime in args.regimes:
        cfg = EnvConfig.from_json(args.config) if args.config else EnvConfig()
        cfg.workload.regime = regime
        if args.episode_length:
            cfg.episode_length = args.episode_length
        name = args.name if (args.name and len(args.regimes) == 1) else regime
        res = run_experiment(name, cfg, outdir_root=args.out, models_dir=args.models_dir,
                             include_drl=not args.no_drl)
        print(res["summary"][["policy", "hotspot_frequency_mean", "peak_temperature_mean",
                              "tasks_completed_mean", "episode_return_mean"]].round(3).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
