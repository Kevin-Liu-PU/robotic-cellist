from dataclasses import dataclass
import numpy as np

@dataclass
class BowModel:
    frog_tool: np.ndarray
    tip_tool: np.ndarray

    def __post_init__(self):
        self.frog_tool = np.asarray(self.frog_tool, dtype=float)
        self.tip_tool = np.asarray(self.tip_tool, dtype=float)
        if self.frog_tool.shape != (3,) or self.tip_tool.shape != (3,):
            raise ValueError("Bow endpoints must each be 3D vectors")
        if np.linalg.norm(self.tip_tool-self.frog_tool) < 1e-9:
            raise ValueError("Bow hair segment must have non-zero length")

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.tip_tool-self.frog_tool))

    @property
    def axis_tool(self) -> np.ndarray:
        return (self.tip_tool-self.frog_tool) / self.length
