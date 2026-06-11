"""Wandb / file-logging helpers. Set WANDB_API_KEY or run `wandb login` to enable wandb."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional


def init_wandb(
    *,
    project: str,
    run_name: str,
    config: Optional[dict] = None,
    workdir: Optional[str | Path] = None,
    enabled: bool = True,
):
    """Initialize wandb if WANDB_API_KEY is set; otherwise silently no-op.

    Returns the wandb ``run`` object, or ``None`` when disabled.
    """
    if not enabled:
        return None
    # Accept WANDB_API_KEY or stored `wandb login` credentials in ~/.netrc.
    netrc = Path.home() / ".netrc"
    have_creds = bool(os.environ.get("WANDB_API_KEY")) or (
        netrc.exists() and "api.wandb.ai" in netrc.read_text(errors="ignore")
    )
    if not have_creds:
        logging.warning(
            "No wandb credentials found; wandb logging disabled. "
            "Run `wandb login` or set WANDB_API_KEY to enable."
        )
        return None
    import wandb
    return wandb.init(
        project=project,
        name=run_name,
        config=config or {},
        dir=str(workdir) if workdir else None,
    )


def setup_logging(
    workdir: Optional[str | Path] = None,
    *,
    level: int = logging.INFO,
) -> logging.Logger:
    """Console + (optional) file logger with consistent format."""
    logger = logging.getLogger("main")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if workdir is not None:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(workdir / "train.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
