"""Shared chart palette (used by the saved figures and the Streamlit dashboard).

Colour follows the *entity* (the scheduling policy), never its rank, so a
filter that removes a policy never repaints the others. The five categorical
hues are slots 1-5 of the validated reference palette (adjacent-pair CVD
separation and normal-vision floor pass in both light and dark modes). The
random baseline is a neutral grey: it is a sanity reference, not a contender.

Light-mode contrast for aqua / yellow / magenta is below 3:1, so every chart
that uses them also ships visible value labels and an accompanying table.
"""

LIGHT = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "ink_muted": "#8a8984",
    "grid": "#e6e5e1",
    "threshold": "#e34948",   # status colour (critical), reserved for limits
}

DARK = {
    "surface": "#0b0f19",
    "card": "#111827",
    "card_border": "rgba(255, 255, 255, 0.08)",
    "ink": "#f1f5f9",
    "ink_secondary": "#94a3b8",
    "ink_muted": "#64748b",
    "grid": "rgba(255, 255, 255, 0.07)",
    "threshold": "#ef4444",
    "safe": "#10b981",
    "warning": "#f59e0b",
    "accent": "#f97316",
    "cyan": "#06b6d4",
}

POLICY_COLORS = {
    "round-robin": "#2a78d6",
    "greedy-utilisation": "#eb6834",
    "greedy-coolest": "#1baf7a",
    "PPO": "#eda100",
    "DQN": "#e87ba4",
    "random": "#8a8984",
}

POLICY_ORDER = ["random", "round-robin", "greedy-utilisation", "greedy-coolest",
                "PPO", "DQN"]

POLICY_LABELS = {
    "random": "Random (valid)",
    "round-robin": "Round-robin",
    "greedy-utilisation": "Greedy utilisation",
    "greedy-coolest": "Greedy coolest",
    "PPO": "PPO (DRL)",
    "DQN": "DQN (DRL)",
}


def base_policy(name: str) -> str:
    """Map e.g. 'PPO-seed0' or 'PPO (mean of 3 seeds)' to its base policy."""
    for key in POLICY_COLORS:
        if name == key or name.startswith(key + "-") or name.startswith(key + " "):
            return key
    return name


def color_for(name: str) -> str:
    return POLICY_COLORS.get(base_policy(name), "#8a8984")


def label_for(name: str) -> str:
    return POLICY_LABELS.get(name, name)


def sort_policies(names):
    order = {k: i for i, k in enumerate(POLICY_ORDER)}
    return sorted(names, key=lambda n: (order.get(base_policy(n), 99), n))
