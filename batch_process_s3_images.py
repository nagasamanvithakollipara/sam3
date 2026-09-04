#!/usr/bin/env python3
"""
Batch process all images in a folder using SAM3
Supports both local folders and S3 folders
"""

import argparse
import os
import sys
import json
import boto3
from pathlib import Path
import torch

# Add SAM3 to path
# Assumes SAM3 is cloned in the same directory (SAM3/sam3) or specified path
SAM3_PATH = os.getenv('SAM3_PATH', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sam3'))
if os.path.exists(SAM3_PATH):
    sys.path.insert(0, SAM3_PATH)

# Import the main script components
from s3_detection_segmentation import SAM3Detector, S3ImageProcessor, save_segmentation_results, save_colored_segmented_images, save_binary_masks


def list_local_images(folder_path):
    """List all image files in a local folder, excluding thumbnails folders"""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
    images = []
    
    # Normalize the folder path
    folder_path = os.path.abspath(os.path.expanduser(folder_path))
    
    if not os.path.isdir(folder_path):
        raise ValueError(f"Input folder does not exist: {folder_path}")
    
    # Walk through the directory
    for root, dirs, files in os.walk(folder_path):
        # Skip thumbnails directories
        dirs[:] = [d for d in dirs if 'thumbnail' not in d.lower()]
        
        for file in files:
            file_lower = file.lower()
            # Check if it's an image file
            if file_lower.endswith(image_extensions):
                full_path = os.path.join(root, file)
                images.append(full_path)
    
    # Sort images for consistent processing order
    images.sort()
    
    return images


def list_s3_images(s3_client, bucket, prefix):
    """List all image files in S3 prefix, excluding thumbnails folders"""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
    images = []
    
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                key_lower = key.lower()
                
                # Skip files in thumbnails folders
                if '/thumbnails/' in key_lower or key_lower.startswith('thumbnails/'):
                    continue
                
                # Check if it's an image file
                if key_lower.endswith(image_extensions):
                    images.append(f's3://{bucket}/{obj["Key"]}')
    
    return images


def main():
    parser = argparse.ArgumentParser(
        description="Batch process all images in a folder using SAM3 (supports local folders and S3)"
    )
    
    parser.add_argument(
        "--input_folder",
        type=str,
        default=None,
        help="Input folder path - can be local folder (e.g., '/path/to/images') or S3 folder (e.g., 's3://bucket-name/folder/')"
    )
    
    # Keep s3_folder_path for backward compatibility
    parser.add_argument(
        "--s3_folder_path",
        type=str,
        default=None,
        help="[DEPRECATED] S3 folder path (e.g., 's3://bucket-name/folder/'). Use --input_folder instead."
    )
    parser.add_argument(
        "--text_prompt",
        type=str,
        required=True,
        help="Text prompt for detection (e.g., 'car . person . dog')"
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="first_match",
        choices=["first_match", "all"],
        help="How to use multi-concept prompts: first_match stops at first successful concept; all merges all concepts."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs1",
        help="Directory to save output images"
    )
    parser.add_argument(
        "--colored_segments_dir",
        type=str,
        default=None,
        help="Directory to save colored segmented image with all masks combined. If not specified, uses output_dir/colored_segments"
    )
    parser.add_argument(
        "--binary_masks_dir",
        type=str,
        default=None,
        help="Directory to save binary masks (black and white). If not specified, uses output_dir/binary_masks"
    )
    parser.add_argument(
        "--output_s3_path",
        type=str,
        default=None,
        help="Optional S3 folder path to upload results (e.g., 's3://bucket-name/output/')"
    )
    
    # Model configuration
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token for accessing SAM3 checkpoints (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for SAM3 detections (default: 0.5)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to run on (cuda or cpu). Default: auto-detect (cuda if available)"
    )
    parser.add_argument(
        "--auto_fallback_to_cpu",
        action="store_true",
        help="Automatically fallback to CPU if GPU memory is insufficient"
    )
    parser.add_argument(
        "--min_gpu_memory_gb",
        type=float,
        default=2.0,
        help="Minimum free GPU memory required in GB (default: 2.0)"
    )
    parser.add_argument(
        "--max_image_side",
        type=int,
        default=1600,
        help="Resize images so max(height,width) <= this before SAM3 (reduces VRAM spikes). Use 0 to disable."
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable CUDA autocast (AMP). By default AMP is enabled to reduce VRAM usage."
    )
    parser.add_argument(
        "--no_clear_cache_each_image",
        action="store_true",
        help="Disable aggressive CUDA cache clearing between images."
    )
    
    # Detection parameters
    parser.add_argument(
        "--box_threshold",
        type=float,
        default=0.3,
        help="Box threshold for filtering detections"
    )
    parser.add_argument(
        "--text_threshold",
        type=float,
        default=0.25,
        help="Text threshold for filtering detections"
    )
    
    # AWS credentials
    parser.add_argument(
        "--aws_access_key_id",
        type=str,
        default=None,
        help="AWS access key ID"
    )
    parser.add_argument(
        "--aws_secret_access_key",
        type=str,
        default=None,
        help="AWS secret access key"
    )
    parser.add_argument(
        "--aws_region",
        type=str,
        default="us-east-1",
        help="AWS region"
    )
    parser.add_argument(
        "--combined_outputs",
        action="store_true",
        help="Write ONE file per input image containing all detections, instead of separate _object_N files."
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process (default: process all)"
    )
    
    args = parser.parse_args()
    
    # Determine input folder path (support both new and old argument names)
    input_folder = args.input_folder
    if input_folder is None and args.s3_folder_path:
        input_folder = args.s3_folder_path
        print("Warning: --s3_folder_path is deprecated. Use --input_folder instead.")
    
    if input_folder is None:
        print("Error: --input_folder is required (or use --s3_folder_path for backward compatibility)")
        parser.print_help()
        return
    
    # Determine if input is S3 or local folder
    is_s3 = input_folder.startswith('s3://')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create overlays directory (original image + mask overlay)
    overlays_dir = os.path.join(args.output_dir, "overlays")
    os.makedirs(overlays_dir, exist_ok=True)
    
    # Create colored segments output directory
    if args.colored_segments_dir is None:
        colored_segments_dir = os.path.join(args.output_dir, "colored_segments")
    else:
        colored_segments_dir = args.colored_segments_dir
    os.makedirs(colored_segments_dir, exist_ok=True)
    
    # Create binary masks output directory
    if args.binary_masks_dir is None:
        binary_masks_dir = os.path.join(args.output_dir, "binary_masks")
    else:
        binary_masks_dir = args.binary_masks_dir
    os.makedirs(binary_masks_dir, exist_ok=True)
    
    # List images based on input type
    if is_s3:
        # S3 path handling
        s3_path = input_folder[5:] if input_folder.startswith('s3://') else input_folder
        parts = s3_path.split('/', 1)
        if len(parts) != 2:
            print(f"Invalid S3 path format: {input_folder}")
            return
        
        bucket_name, folder_prefix = parts
        if not folder_prefix.endswith('/'):
            folder_prefix += '/'
        
        # Initialize S3 processor
        s3_processor = S3ImageProcessor(
            aws_access_key_id=args.aws_access_key_id,
            aws_secret_access_key=args.aws_secret_access_key,
            aws_region=args.aws_region
        )
        
        # List all images in S3 folder
        print(f"Listing images in s3://{bucket_name}/{folder_prefix}...")
        s3_client = boto3.client(
            's3',
            aws_access_key_id=args.aws_access_key_id,
            aws_secret_access_key=args.aws_secret_access_key,
            region_name=args.aws_region
        )
        
        image_paths = list_s3_images(s3_client, bucket_name, folder_prefix)
        
        if not image_paths:
            print("No images found in the specified S3 folder!")
            return
    else:
        # Local folder handling
        print(f"Listing images in local folder: {input_folder}...")
        try:
            image_paths = list_local_images(input_folder)
        except Exception as e:
            print(f"Error listing images from local folder: {e}")
            return
        
        if not image_paths:
            print("No images found in the specified folder!")
            return
        
        # S3 processor not needed for local folders, but initialize it anyway for optional upload
        s3_processor = None
        if args.output_s3_path:
            s3_processor = S3ImageProcessor(
                aws_access_key_id=args.aws_access_key_id,
                aws_secret_access_key=args.aws_secret_access_key,
                aws_region=args.aws_region
            )
    
    # Limit to max_images if specified
    total_images = len(image_paths)
    if args.max_images is not None and args.max_images > 0:
        image_paths = image_paths[:args.max_images]
        print(f"Found {total_images} images, processing first {len(image_paths)} images...")
    else:
        print(f"Processing {len(image_paths)} images...")
    
    # Use provided token, env var, or default
    hf_token = args.hf_token or os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN') or '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'
    
    # Determine device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Initialize SAM3 detector (once for all images)
    try:
        detector = SAM3Detector(
            hf_token=hf_token,
            device=device,
            confidence_threshold=args.confidence_threshold,
            auto_fallback_to_cpu=args.auto_fallback_to_cpu,
            min_gpu_memory_gb=args.min_gpu_memory_gb,
            max_image_side=args.max_image_side,
            use_amp=(not args.no_amp),
            clear_cache_each_image=(not args.no_clear_cache_each_image),
        )
    except Exception as e:
        print(f"Error initializing SAM3 model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Process each image
    results_summary = []
    for i, image_path in enumerate(image_paths, 1):
        
        try:
            # Determine if this is an S3 path or local path
            is_s3_image = image_path.startswith('s3://')
            
            if is_s3_image:
                # Download image from S3 to temporary location
                import tempfile
                temp_dir = tempfile.mkdtemp()
                input_filename = os.path.basename(image_path)
                local_input_path = os.path.join(temp_dir, input_filename)
                
                s3_processor.download_image(image_path, local_input_path)
                
                # Store temp_dir for cleanup
                temp_dir_to_clean = temp_dir
                cleanup_temp = True
            else:
                # Use local image directly
                local_input_path = image_path
                input_filename = os.path.basename(image_path)
                cleanup_temp = False
                temp_dir_to_clean = None
            
            # Perform detection and segmentation
            annotated_image, masks, boxes, logits, phrases, processing_time = detector.detect_and_segment(
                image_path=local_input_path,
                text_prompt=args.text_prompt,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                prompt_mode=args.prompt_mode,
            )
            
            # Save results (use original filename)
            output_filename = input_filename
            # Overlay image (original + mask) -> outputs/overlays/
            overlay_path = os.path.join(overlays_dir, output_filename)
            save_segmentation_results(
                annotated_image,
                masks,
                boxes,
                phrases,
                overlay_path,
                image_path=local_input_path,
                scores=logits,
                combined=args.combined_outputs,
            )
            
            # Save colored segmented images per region (like vinay.py format)
            save_colored_segmented_images(local_input_path, masks, colored_segments_dir, image_name=output_filename, combined=args.combined_outputs)
            
            # Save binary masks per instance (one file per object, like overlays and colored segments)
            save_binary_masks(masks, binary_masks_dir, image_name=output_filename, combined=args.combined_outputs)
            
            # Clean up temporary input file if downloaded from S3
            if cleanup_temp and temp_dir_to_clean:
                try:
                    if os.path.exists(local_input_path):
                        os.remove(local_input_path)
                    if os.path.exists(temp_dir_to_clean):
                        os.rmdir(temp_dir_to_clean)
                except:
                    pass
            
            print(f"[{i}/{len(image_paths)}] {input_filename}: {len(boxes)} objects ({processing_time:.2f}s)")
            
            # Upload to S3 if specified (upload overlay image)
            if args.output_s3_path and s3_processor:
                output_s3_full_path = args.output_s3_path.rstrip('/') + '/overlays/' + output_filename
                try:
                    s3_processor.upload_image(overlay_path, output_s3_full_path)
                except Exception as e:
                    print(f"  Error uploading to S3: {e}")
            
            results_summary.append({
                'image': image_path,
                'objects_found': len(boxes),
                'phrases': phrases,
                'scores': [float(x) for x in (logits.tolist() if hasattr(logits, "tolist") else list(logits))] if len(boxes) > 0 else [],
                'processing_time': processing_time,
                'success': True
            })
            
        except Exception as e:
            print(f"  ERROR processing image: {e}")
            results_summary.append({
                'image': image_path,
                'success': False,
                'error': str(e)
            })
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    successful = sum(1 for r in results_summary if r['success'])
    if args.max_images is not None:
        print(f"Total images found: {total_images}")
        print(f"Images processed: {len(image_paths)}")
    else:
        print(f"Total images: {len(image_paths)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(image_paths) - successful}")
    
    # Calculate total and average processing time
    successful_results = [r for r in results_summary if r['success']]
    if successful_results:
        total_time = sum(r.get('processing_time', 0) for r in successful_results)
        avg_time = total_time / len(successful_results)
        print(f"\nTotal processing time: {total_time:.2f} seconds")
        print(f"Average processing time per image: {avg_time:.2f} seconds")

    # Write machine-readable summary with per-image scores for downstream scripts.
    summary_path = os.path.join(args.output_dir, "predictions_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"Wrote predictions summary: {summary_path}")
    
    print("\nResults saved to:", args.output_dir)


if __name__ == "__main__":
    main()

