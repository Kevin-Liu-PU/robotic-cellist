from dataclasses import dataclass
import numpy as np

@dataclass
class StringModel:
    id: str
    bridge_pt: np.ndarray
    far_pt: np.ndarray
    normal: np.ndarray

    def __post_init__(self):
        self.bridge_pt = np.asarray(self.bridge_pt, dtype=float)
        self.far_pt = np.asarray(self.far_pt, dtype=float)
        self.normal = np.asarray(self.normal, dtype=float)
        if any(x.shape != (3,) for x in (self.bridge_pt, self.far_pt, self.normal)):
            raise ValueError("String geometry vectors must be 3D")
        if np.linalg.norm(self.far_pt-self.bridge_pt) < 1e-9:
            raise ValueError(f"String {self.id} has zero length")
        n = np.linalg.norm(self.normal)
        if n < 1e-9:
            raise ValueError(f"String {self.id} normal cannot be zero")
        self.normal = self.normal / n

    @property
    def tangent(self):
        v = self.far_pt-self.bridge_pt
        return v / np.linalg.norm(v)

@dataclass
class CelloModel:
    strings: list[StringModel]
    fiducials: dict[str, np.ndarray]

    def __post_init__(self):
        if len(self.strings) != 4:
            raise ValueError("Expected four cello strings")
        self.fiducials = {k: np.asarray(v, dtype=float) for k, v in self.fiducials.items()}
        if any(v.shape != (3,) for v in self.fiducials.values()):
            raise ValueError("Each fiducial must be a 3D point")
