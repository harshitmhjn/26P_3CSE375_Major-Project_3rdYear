"""Step-by-step simulation session used by the dashboard (and usable from scripts).

``LiveSimulation`` wraps one environment and one policy and records a per-step
history, so a UI can advance the simulation a few steps at a time, pause, and
reset -- while every number shown comes from the same environment the
experiment pipeline uses. No Streamlit import here.
"""

from .config import EnvConfig
from .env import DataCenterSchedulingEnv


class LiveSimulation:
    def __init__(self, cfg: EnvConfig, policy, seed: int = None):
        self.cfg = cfg
        self.policy = policy
        self.seed = cfg.seed if seed is None else seed
        self.reset()

    # -- lifecycle ---------------------------------------------------------

    def reset(self, seed: int = None):
        if seed is not None:
            self.seed = seed
        self.env = DataCenterSchedulingEnv(self.cfg)
        self.env.reset(seed=self.seed)
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        self.history = []
        self.cumulative_reward = 0.0
        self.done = False
        self.terminated = False

    def set_policy(self, policy):
        """Swap the scheduling policy mid-run (the environment state is kept)."""
        self.policy = policy
        if hasattr(policy, "reset"):
            policy.reset()

    # -- stepping ----------------------------------------------------------

    def step(self, n: int = 1) -> int:
        """Advance up to ``n`` steps. Returns the number actually executed."""
        executed = 0
        for _ in range(max(int(n), 0)):
            if self.done:
                break
            action = self.policy.act(self.env)
            _, reward, term, trunc, info = self.env.step(action)
            self.cumulative_reward += reward
            temps = self.env.temps
            self.history.append({
                "step": self.env.step_count,
                "mean_temp": float(temps.mean()),
                "max_temp": float(temps.max()),
                "min_temp": float(temps.min()),
                "hotspot": bool(temps.max() > self.cfg.thermal.safe_temp),
                "queue": len(self.env.queue),
                "running": sum(len(r) for r in self.env.running),
                "completed": self.env.stats.completed,
                "dropped": self.env.stats.dropped,
                "it_power_kw": info["it_power_kw"],
                "cooling_power_kw": info["cooling_power_kw"],
                "utilisation": float(self.env.alloc.mean() / self.cfg.facility.server_capacity),
                "reward": float(reward),
                "action": int(action),
            })
            executed += 1
            if term or trunc:
                self.done = True
                self.terminated = bool(term)
        return executed

    def run_to_end(self) -> int:
        return self.step(self.cfg.episode_length - self.env.step_count + 1)

    # -- read access -------------------------------------------------------

    @property
    def step_count(self) -> int:
        return self.env.step_count

    def summary(self) -> dict:
        s = self.env.stats.summary()
        s["episode_return_so_far"] = self.cumulative_reward
        return s

    def snapshot(self) -> list:
        return self.env.snapshot()

    def queue_view(self) -> list:
        return [{"task_id": t.task_id, "cpu_demand": round(t.demand, 3),
                 "duration": t.duration, "category": t.category,
                 "waiting_steps": self.env.step_count - t.arrival_step}
                for t in self.env.queue]
