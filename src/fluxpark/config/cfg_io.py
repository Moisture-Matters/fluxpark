import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Union

from .fluxpark_config import FluxParkConfig

CFG_FILENAME = "fluxpark_cfg.json"


def save_cfg(cfg: FluxParkConfig, outdir: Union[str, Path]) -> Path:
    """Serialize a FluxParkConfig to JSON in outdir."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / CFG_FILENAME
    data = asdict(cfg)
    # Convert Path objects to strings so JSON can handle them
    for key, val in data.items():
        if isinstance(val, Path):
            data[key] = str(val)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def load_cfg(outdir: Union[str, Path]) -> FluxParkConfig:
    """Load a FluxParkConfig from a JSON file in outdir."""
    path = Path(outdir) / CFG_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"No configuration file found at '{path}'. "
            "Run the model with eval_waterbalance=True first, or provide the "
            "configuration manually."
        )
    with open(path) as f:
        data = json.load(f)
    known = {f.name for f in fields(FluxParkConfig)}
    return FluxParkConfig(**{k: v for k, v in data.items() if k in known})
