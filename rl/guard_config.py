import math


GUARD_WEIGHTS = (
    "W_DYNAMIC", "W_TIMING", "W_BOW", "W_SMOOTH", "W_ONSET_ACCEL",
    "W_ENVELOPE", "W_SPEAK", "W_SPEED_TARGET", "W_DEFECT",
)
REWARD_WEIGHTS = ("W_QUALITY", *GUARD_WEIGHTS, "W_SPEAK_MEAN")


def parse_guard_mode(value=None):
    if value is None:
        return "full"
    if value not in ("full", "off"):
        raise ValueError("CELLO_GUARD_MODE must be exactly 'full' or 'off'")
    return value


def apply_guard_mode(namespace, mode):
    mode = parse_guard_mode(mode)
    if mode == "off":
        if namespace["W_QUALITY"] != 0.50:
            raise ValueError("guard-off requires W_QUALITY=0.50; cannot combine with ablation D")
        for name in GUARD_WEIGHTS:
            namespace[name] = 0.0
        namespace["W_SPEAK_MEAN"] = 0.0


def guard_configuration(namespace, mode):
    mode = parse_guard_mode(mode)
    weights = {name: float(namespace[name]) for name in REWARD_WEIGHTS}
    if not all(math.isfinite(value) for value in weights.values()):
        raise ValueError("nonfinite reward weight")
    if mode == "off" and (
        weights["W_QUALITY"] != 0.50
        or any(weights[name] != 0.0 for name in (*GUARD_WEIGHTS, "W_SPEAK_MEAN"))
    ):
        raise ValueError("guard-off reward configuration was overridden")
    return {
        "schema": "G1-guard-off-v1", "mode": mode,
        "effective_weights": weights,
        "speed_prior_effective_weight": "W_SPEED_TARGET * (1 - window_fill)",
        "defect_origin": "handcrafted acoustic features and thresholds, not learned heads",
        "quality_eff_unchanged": True, "renormalized": False,
        "safety_constraints_unchanged": True,
    }
