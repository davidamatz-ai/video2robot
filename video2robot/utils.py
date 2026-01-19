"""Shared utilities for video2robot."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional, Any

from video2robot.config import DATA_DIR


def load_smplx_file_robust(smplx_file: str | Path, smplx_body_model_path: str):
    """
    Robust version of load_smplx_file that handles mismatched beta sizes.
    """
    import numpy as np
    import smplx
    import torch

    smplx_data = np.load(smplx_file, allow_pickle=True)

    # Get number of betas from data
    if "betas" in smplx_data:
        betas_data = smplx_data["betas"]
        num_betas = betas_data.shape[-1] if len(betas_data.shape) > 0 else 1
    else:
        num_betas = 10

    # Create body model with correct number of betas
    body_model = smplx.create(
        smplx_body_model_path,
        "smplx",
        gender=str(smplx_data["gender"]),
        use_pca=False,
        num_betas=num_betas,
    )

    num_frames = smplx_data["pose_body"].shape[0]

    # betas: repeat single vector to batch size
    betas = torch.tensor(smplx_data["betas"]).float().view(1, -1).repeat(num_frames, 1)

    # expression: use from file if available, otherwise zeros
    if "expression" in smplx_data:
        expression = torch.tensor(smplx_data["expression"]).float()
    else:
        expression = torch.zeros(num_frames, 10).float()

    # Ensure expression matches model expectations if specified
    # Default SMPL-X has 10 expression coeffs
    if hasattr(body_model, 'num_expression_coeffs') and expression.shape[-1] != body_model.num_expression_coeffs:
        if expression.shape[-1] < body_model.num_expression_coeffs:
            padding = torch.zeros(num_frames, body_model.num_expression_coeffs - expression.shape[-1])
            expression = torch.cat([expression, padding], dim=-1)
        else:
            expression = expression[:, :body_model.num_expression_coeffs]

    smplx_output = body_model(
        betas=betas,
        global_orient=torch.tensor(smplx_data["root_orient"]).float(),
        body_pose=torch.tensor(smplx_data["pose_body"]).float(),
        transl=torch.tensor(smplx_data["trans"]).float(),
        left_hand_pose=torch.zeros(num_frames, 45).float(),
        right_hand_pose=torch.zeros(num_frames, 45).float(),
        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        expression=expression,
        return_full_pose=True,
    )

    if len(smplx_data["betas"].shape) == 1:
        human_height = 1.66 + 0.1 * smplx_data["betas"][0]
    else:
        human_height = 1.66 + 0.1 * smplx_data["betas"][0, 0]

    return smplx_data, body_model, smplx_output, human_height


def emit_progress(stage: str, value: float, message: str, **kwargs):
    """Emit a standardized progress marker for TaskManager parsing.

    Format: [Progress] stage=<name> value=<0.0-1.0> message=<text> [key=value ...]

    Args:
        stage: Stage identifier (e.g., "init", "generating", "done")
        value: Progress value between 0.0 and 1.0
        message: Human-readable status message
        **kwargs: Additional key-value pairs (e.g., frames="100/200")
    """
    value = max(0.0, min(1.0, value))
    parts = [f"[Progress] stage={stage} value={value:.2f} message={message}"]
    for k, v in kwargs.items():
        parts.append(f"{k}={v}")
    print(" ".join(parts), flush=True)


def get_next_project_dir(prefix: str = "video") -> Path:
    """Get next available project directory (video_001, video_002, ...)"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(DATA_DIR.glob(f"{prefix}_*"))

    if not existing:
        num = 1
    else:
        nums = []
        for d in existing:
            try:
                nums.append(int(d.name.split("_")[-1]))
            except ValueError:
                continue
        num = max(nums) + 1 if nums else 1

    return DATA_DIR / f"{prefix}_{num:03d}"


def ensure_project_dir(
    project_path: Optional[str | Path] = None,
    name: Optional[str] = None,
) -> Path:
    """Ensure project directory exists and return its path.

    Args:
        project_path: Explicit project path (takes priority)
        name: Folder name under DATA_DIR (used if project_path is None)

    Returns:
        Resolved project directory path

    Examples:
        >>> ensure_project_dir("/path/to/project")  # Use explicit path
        >>> ensure_project_dir(name="my_project")   # DATA_DIR/my_project
        >>> ensure_project_dir()                    # Auto-generate video_XXX
    """
    if project_path:
        path = Path(project_path)
    elif name:
        path = DATA_DIR / name
    else:
        path = get_next_project_dir()

    path.mkdir(parents=True, exist_ok=True)
    return path


def run_in_conda(env_name: str, argv: list[str], cwd: Path, *, raise_on_error: bool = True):
    """Run a command inside a conda environment.

    Args:
        env_name: Conda environment name
        argv: Command arguments
        cwd: Working directory
        raise_on_error: If True, raise RuntimeError on failure; if False, print error
    """
    cmd = ["conda", "run", "--no-capture-output", "-n", env_name, *argv]
    try:
        result = subprocess.run(cmd, cwd=str(cwd))
    except KeyboardInterrupt:
        if raise_on_error:
            raise RuntimeError("Interrupted by user.") from None
        print("\n[Info] Interrupted.")
        return
    if result.returncode != 0:
        msg = f"Command failed (env={env_name}): {' '.join(argv)}"
        if raise_on_error:
            raise RuntimeError(msg)
        print(f"[Error] {msg}")
