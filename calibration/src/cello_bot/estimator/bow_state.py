"""Offline geometric bow-state estimator."""
from dataclasses import dataclass
import numpy as np
from cello_bot.geometry.transforms import transform_points, transform_direction, invert_transform
from cello_bot.geometry.segments import closest_points_segments

@dataclass
class BowState:
    string_id: str | None
    bow_fraction: float
    string_fraction: float
    bow_point_cello: np.ndarray
    string_point_cello: np.ndarray
    gap: float
    active: bool

class BowStateEstimator:
    def __init__(self, bow, cello, T_base_cello, contact_tolerance=0.003):
        self.bow = bow
        self.cello = cello
        self.T_base_cello = np.asarray(T_base_cello, dtype=float)
        self.contact_tolerance = float(contact_tolerance)

    def estimate(self, T_base_tool):
        T_cello_base = invert_transform(self.T_base_cello)
        T_cello_tool = T_cello_base @ np.asarray(T_base_tool, dtype=float)
        hair = transform_points(T_cello_tool, np.vstack([self.bow.frog_tool, self.bow.tip_tool]))
        best = None
        for string in self.cello.strings:
            a, b, u, v, dist = closest_points_segments(
                hair[0], hair[1], string.bridge_pt, string.far_pt
            )
            # Signed gap is diagnostic: positive/negative side relative to string normal.
            signed_gap = float((a-b) @ string.normal)
            candidate = (dist, string, a, b, u, v, signed_gap)
            if best is None or dist < best[0]:
                best = candidate
        dist, string, a, b, u, v, signed_gap = best
        return BowState(
            string_id=string.id if dist <= self.contact_tolerance else None,
            bow_fraction=float(u),
            string_fraction=float(v),
            bow_point_cello=a,
            string_point_cello=b,
            gap=signed_gap,
            active=bool(dist <= self.contact_tolerance),
        )
