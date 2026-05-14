import argparse
import pickle
import os
import sys

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GMR pickle files to CSV (for beyondmimic)")
    parser.add_argument(
        "--folder", type=str, help="Path to the folder containing pickle files from GMR",
    )
    args = parser.parse_args()

    input_file = os.path.join(args.folder, "robot_motion.pkl")
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        sys.exit(1)

    out_folder = os.path.join(args.folder, "csv")
    os.makedirs(out_folder, exist_ok=True)

    with open(input_file, "rb") as f:
        motion_data = pickle.load(f)

    dof_pos = motion_data["dof_pos"]
    frame_rate = motion_data["fps"]
    motion = np.zeros((dof_pos.shape[0], dof_pos.shape[1] + 7), dtype=np.float32)
    motion[:, :3] = motion_data["root_pos"]
    motion[:, 3:7] = motion_data["root_rot"]
    motion[:, 7:] = dof_pos

    if frame_rate > 30:
        # downsample to 30 fps
        downsample_factor = frame_rate / 30.0
        indices = np.arange(0, motion.shape[0], downsample_factor).astype(int)
        old_length = motion.shape[0]
        motion = motion[indices]
        print(f"Downsampled from {old_length} to {motion.shape[0]} frames")


    output_path = os.path.join(out_folder, "robot_motion.csv")
    np.savetxt(
        output_path,
        motion,
        delimiter=",",
    )
    print(f"Saved to {output_path}")
