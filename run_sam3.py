#!/usr/bin/env python3
"""
Simplified launcher for equirect_multiview_sam3.py.

Just give it an S3 image folder and a text prompt; every other flag
uses the sensible defaults below. Any default can still be overridden
by passing the corresponding --flag through (they take priority).

Usage:
    python run_sam3.py "s3://bucket/path/to/images" "safety helmet . person . hat . sky"

    # override anything by adding extra flags, e.g.:
    python run_sam3.py "s3://bucket/path" "person . hat" --max_images 5 --output_dir ./out2
"""
import subprocess
import sys
import os

from dotenv import load_dotenv

# Load AWS credentials (and anything else) from a .env file next to this
# script, e.g.:
#   AWS_ACCESS_KEY_ID=...
#   AWS_SECRET_ACCESS_KEY=...
#   AWS_SESSION_TOKEN=...      (optional, for temporary/SSO creds)
#   AWS_DEFAULT_REGION=us-east-1
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

DEFAULTS = {
    "--prompt_mode": "all",
    "--views": "0:-20,72:-20,144:-20,216:-20,288:-20,36:50,156:50,276:50,0:-90,0:90",
    "--fov": "140",
    "--concept_views": "safety helmet=-90:55,person=-90:55,hat=-90:55",
    "--concept_threshold": "sky=0.85",
    "--merge_size_ratio": "1.5",
    "--model_resolution": "336",
    "--mask_close": "5",
    "--aws_region": "us-east-1",
    "--output_dir": "./outputs_equirect",
}

# Flags with no value (booleans) that are on by default.
DEFAULT_FLAGS = [
    "--black_overlay",
    "--fill_holes",
    # --save_views intentionally left out: it writes every per-view crop/mask
    # to disk (a "views" subfolder) and measurably slows down each frame.
    # Pass --save_views explicitly if you need those debug images.
]

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "equirect_multiview_sam3.py")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    input_folder = sys.argv[1]
    text_prompt = sys.argv[2]
    extra_args = sys.argv[3:]

    cmd = [sys.executable, SCRIPT,
           "--input_folder", input_folder,
           "--text_prompt", text_prompt]

    # Add defaults, but let anything already present in extra_args win.
    for flag, value in DEFAULTS.items():
        if flag not in extra_args:
            cmd += [flag, value]
    for flag in DEFAULT_FLAGS:
        if flag not in extra_args:
            cmd.append(flag)

    cmd += extra_args

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
