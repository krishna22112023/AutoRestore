import yaml
from pathlib import Path
from functools import lru_cache
import copy
from typing import Any, Dict


@lru_cache(maxsize=None)
def load_config(experiment: str = "base") -> Dict[str, Any]:
    """Load *config/base.yaml* and return the configuration for a given experiment.

    The YAML file is structured as:

    experiments:
      base: &base_template
        ...
      my_exp:
        <<: *base_template
        ... overrides ...

    We replicate the YAML anchor/alias behaviour in Python by performing a deep
    merge of the chosen experiment onto the *base* template. Returned value is
    a plain dict that callers may freely mutate.
    """
    cfg_path = Path(__file__).resolve().parent / "base.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    experiments = raw.get("experiments", {})
    if "base" not in experiments:
        raise KeyError("base.yaml must define a 'base' experiment template under experiments.base")

    base_cfg = experiments["base"]
    if experiment == "base":
        return copy.deepcopy(base_cfg)

    if experiment not in experiments:
        raise KeyError(f"Experiment '{experiment}' not found in base.yaml")

    exp_cfg = experiments[experiment]

    # Deep copy base and recursively update with experiment overrides
    merged = copy.deepcopy(base_cfg)

    def _deep_update(target: Dict[str, Any], updates: Dict[str, Any]):
        for key, value in updates.items():
            if (
                isinstance(value, dict)
                and isinstance(target.get(key), dict)
            ):
                _deep_update(target[key], value)
            else:
                target[key] = copy.deepcopy(value)

    _deep_update(merged, exp_cfg)
    return merged
