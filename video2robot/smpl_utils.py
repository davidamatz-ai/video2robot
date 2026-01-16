import numpy as np
import smplx
import torch

def load_smplx_file_robust(smplx_file, smplx_body_model_path):
    """
    Robust version of load_smplx_file that handles mismatched beta sizes.
    """
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
