# sam3

Scripts for running SAM3-based segmentation (including equirectangular multi-view
detection) and producing binary masks / IoU comparisons.

## Setup

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3 && pip install -e . && cd ..
pip install -r requirements.txt
export HF_TOKEN=your_token_here
```

Place the cloned `sam3` repo in the same directory as these scripts (or pass its
path where the scripts expect `SAM3_PATH`).

## Scripts

- `equirect_multiview_sam3.py` – reprojects an equirectangular image into multiple
  perspective views, runs SAM3 detection/segmentation on each, and merges results
  back onto the equirectangular frame.
- `s3_detection_segmentation.py` – core SAM3 detector/processor (`SAM3Detector`,
  `S3ImageProcessor`) plus helpers for saving segmentation results, colored
  overlays, and binary masks.
- `batch_process_s3_images.py` – batch-runs detection/segmentation over a folder
  of local or S3 images.
