from pathlib import Path
import yaml
import numpy as np
from .bow import BowModel
from .cello import CelloModel, StringModel

def load_models(path):
    """Load BowModel and CelloModel from nominal YAML."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    bow_data = data["bow"]
    cello_data = data["cello"]
    bow = BowModel(
        frog_tool=np.array(bow_data["frog_tool"], dtype=float),
        tip_tool=np.array(bow_data["tip_tool"], dtype=float),
    )
    strings = [
        StringModel(
            id=item["id"],
            bridge_pt=np.array(item["bridge_pt"], dtype=float),
            far_pt=np.array(item["far_pt"], dtype=float),
            normal=np.array(item["normal"], dtype=float),
        )
        for item in cello_data["strings"]
    ]
    fiducials = {k: np.array(v, dtype=float) for k, v in cello_data["fiducials"].items()}
    return bow, CelloModel(strings=strings, fiducials=fiducials)
