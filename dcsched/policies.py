"""Policy registry: heuristic baselines plus any trained DRL checkpoints on disk.

Only checkpoints whose training *completed* are offered; an interrupted run is
never presented as a trained agent.
"""

import os

from .baselines import BASELINES, make_baseline

DEFAULT_MODEL_DIRS = ("results/models", "results/models_ui")


def heuristic_names():
    return ["round-robin", "greedy-utilisation", "greedy-coolest", "random"]


def iter_model_dirs(models_dir: str):
    """The folder itself plus any immediate sub-folders that hold training runs
    (the dashboard writes each UI-launched run into its own sub-folder)."""
    if not os.path.isdir(models_dir):
        return
    yield models_dir
    for name in sorted(os.listdir(models_dir)):
        sub = os.path.join(models_dir, name)
        if os.path.isdir(sub) and any(f.endswith("_progress.json") for f in os.listdir(sub)):
            yield sub


def drl_options(models_dirs=DEFAULT_MODEL_DIRS, root: str = "."):
    """Map label -> run record for every completed, loadable DRL checkpoint."""
    from .training import list_runs
    opts = {}
    for d in models_dirs:
        full = d if os.path.isabs(d) else os.path.join(root, d)
        for sub in iter_model_dirs(full):
            for rec in list_runs(sub):
                if rec.get("status") != "complete" or not rec.get("has_checkpoint"):
                    continue
                label = f"{rec['algo']}-seed{rec['seed']}"
                if label in opts:                   # same label in two folders
                    label += f" [{os.path.basename(sub.rstrip('/'))}]"
                rec["run_dir"] = sub
                opts[label] = rec
    return opts


def build_policy(name: str, drl: dict = None, cfg=None):
    """Construct a policy by name. DRL labels are looked up in ``drl``
    (the dict returned by :func:`drl_options`). If ``cfg`` is given, a DRL
    checkpoint trained for a different facility size is rejected with a clear
    error instead of failing deep inside the network."""
    if name in BASELINES:
        return make_baseline(name)
    if drl and name in drl:
        from .agent import load_model, PolicyAgent
        rec = drl[name]
        model = load_model(rec["algo"], rec["model_path"])
        if cfg is not None:
            want = 2 * cfg.facility.n_servers + 5
            have = int(model.observation_space.shape[0])
            if want != have:
                raise ValueError(
                    f"{name} was trained for a facility with {(have - 5) // 2} servers "
                    f"(observation size {have}); the current configuration has "
                    f"{cfg.facility.n_servers} servers (needs {want}). Choose a matching "
                    f"configuration or train a new agent.")
        return PolicyAgent(model, name=name)
    raise ValueError(f"unknown policy {name!r}: not a baseline and no trained checkpoint with that label")
