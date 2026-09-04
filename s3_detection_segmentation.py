#!/usr/bin/env python3
"""
SAM3 Detection and Segmentation Script for S3 Images

This script downloads images from S3, performs detection and segmentation
using SAM3 based on text prompts, and saves the results.
"""

import argparse
import os
import sys
import boto3
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageFile
import torch
import time
import gc
import contextlib

# Allow PIL to load truncated images (common with corrupted/incomplete files)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Add SAM3 repository to path
# Assumes SAM3 is cloned in the same directory (SAM3/sam3) or specified path
SAM3_PATH = os.getenv('SAM3_PATH', os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    'sam3'
))

if os.path.exists(SAM3_PATH):
    sys.path.insert(0, SAM3_PATH)
else:
    print(f"Warning: SAM3 repository not found at {SAM3_PATH}")
    print("Please clone the repository:")
    print("  git clone https://github.com/facebookresearch/sam3.git")
    print("Or set SAM3_PATH environment variable to the repository path")

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    import supervision as sv
    SAM3_AVAILABLE = True
except ImportError as e:
    SAM3_AVAILABLE = False
    build_sam3_image_model = None
    Sam3Processor = None
    print(f"Warning: Could not import from SAM3: {e}")
    print("Please ensure SAM3 repository is cloned and dependencies are installed:")
    print("  git clone https://github.com/facebookresearch/sam3.git")
    print("  cd sam3")
    print("  pip install -e .")


def check_gpu_memory(min_free_gb=2.0):
    """
    Check if GPU has sufficient free memory.
    Uses pynvml if available to get actual GPU memory (including other processes),
    otherwise falls back to PyTorch's view of memory.
    
    Args:
        min_free_gb: Minimum free GPU memory required in GB
        
    Returns:
        tuple: (has_sufficient_memory: bool, free_memory_gb: float, total_memory_gb: float, message: str)
    """
    if not torch.cuda.is_available():
        return False, 0.0, 0.0, "CUDA is not available"
    
    try:
        # Try to use pynvml for accurate GPU memory info (includes other processes)
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory = mem_info.total / (1024**3)  # GB
            used_memory = mem_info.used / (1024**3)  # GB
            free_memory = mem_info.free / (1024**3)  # GB
            pynvml.nvmlShutdown()
            
            # Clear PyTorch cache
            torch.cuda.empty_cache()
            gc.collect()
            
            has_sufficient = free_memory >= min_free_gb
            
            message = (
                f"GPU Memory Status (from nvidia-smi):\n"
                f"  Total: {total_memory:.2f} GB\n"
                f"  Used: {used_memory:.2f} GB\n"
                f"  Free: {free_memory:.2f} GB\n"
                f"  Required: {min_free_gb:.2f} GB\n"
            )
            
            if not has_sufficient:
                message += (
                    f"\nWARNING: Insufficient GPU memory! "
                    f"Need at least {min_free_gb:.2f} GB free, but only {free_memory:.2f} GB available.\n"
                    f"Other processes are using GPU memory. Check with: nvidia-smi\n"
                )
            
            return has_sufficient, free_memory, total_memory, message
            
        except ImportError:
            # pynvml not available, use PyTorch's view (less accurate)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)  # GB
            reserved_memory = torch.cuda.memory_reserved(0) / (1024**3)  # GB
            
            # Clear cache to get actual free memory
            torch.cuda.empty_cache()
            gc.collect()
            
            # Re-check after clearing cache
            allocated_memory_after = torch.cuda.memory_allocated(0) / (1024**3)
            reserved_memory_after = torch.cuda.memory_reserved(0) / (1024**3)
            free_memory_after = total_memory - reserved_memory_after
            
            # Try to allocate a small tensor to test actual availability
            # This is a more accurate check when pynvml is not available
            try:
                test_tensor = torch.zeros(100, 100, device='cuda')
                del test_tensor
                torch.cuda.empty_cache()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    free_memory_after = 0.0
            
            has_sufficient = free_memory_after >= min_free_gb
            
            message = (
                f"GPU Memory Status (PyTorch view - may not include other processes):\n"
                f"  Total: {total_memory:.2f} GB\n"
                f"  Reserved: {reserved_memory_after:.2f} GB\n"
                f"  Allocated: {allocated_memory_after:.2f} GB\n"
                f"  Free (estimated): {free_memory_after:.2f} GB\n"
                f"  Required: {min_free_gb:.2f} GB\n"
                f"  Note: Install pynvml for accurate GPU memory info: pip install pynvml\n"
            )
            
            if not has_sufficient:
                message += (
                    f"\nWARNING: Insufficient GPU memory! "
                    f"Need at least {min_free_gb:.2f} GB free, but only {free_memory_after:.2f} GB available.\n"
                    f"Other processes may be using GPU memory. Check with: nvidia-smi\n"
                )
            
            return has_sufficient, free_memory_after, total_memory, message
        
    except Exception as e:
        return False, 0.0, 0.0, f"Error checking GPU memory: {e}"


def clear_gpu_memory():
    """Clear GPU memory cache and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # Synchronize to ensure all operations are complete
        torch.cuda.synchronize()
    gc.collect()


def get_gpu_processes():
    """Get list of processes using GPU."""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            processes = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 3:
                        try:
                            pid = int(parts[0].strip())
                            name = parts[1].strip()
                            memory_mb = parts[2].strip()
                            processes.append({'pid': pid, 'name': name, 'memory_mb': memory_mb})
                        except:
                            pass
            return processes
    except:
        pass
    return []


def list_gpu_processes():
    """List processes using GPU and return formatted message."""
    processes = get_gpu_processes()
    if not processes:
        return "No other processes detected using GPU."
    
    msg = f"Found {len(processes)} process(es) using GPU:\n"
    for proc in processes:
        msg += f"  PID {proc['pid']}: {proc['name']} (Memory: {proc['memory_mb']} MB)\n"
    msg += "\nTo free GPU memory, you can:\n"
    msg += "  1. Kill processes: kill -9 <PID>\n"
    msg += "  2. Or wait for them to finish\n"
    return msg


class SAM3Detector:
    """SAM3 Detection and Segmentation Handler"""
    
    def __init__(self, 
                 hf_token=None,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 confidence_threshold=0.5,
                 min_gpu_memory_gb=2.0,
                 auto_fallback_to_cpu=False,
                 max_image_side: int = 1600,
                 use_amp: bool = True,
                 clear_cache_each_image: bool = True,
                 model_resolution: int = 1008):
        """
        Initialize SAM3 model
        
        Args:
            hf_token: Hugging Face token for accessing checkpoints (or set HF_TOKEN env var)
            device: Device to run on (cuda or cpu)
            confidence_threshold: Confidence threshold for filtering detections
            min_gpu_memory_gb: Minimum free GPU memory required in GB (default: 2.0)
            auto_fallback_to_cpu: If True, automatically fallback to CPU if GPU memory is insufficient
            max_image_side: Resize images so max(height,width) <= this before SAM3 (reduces VRAM spikes). None/0 disables.
            use_amp: Enable CUDA autocast (reduces VRAM and can speed up inference).
            clear_cache_each_image: Aggressively clear CUDA cache between images (helps fragmentation in long runs).
            model_resolution: Square size every image is resized to before the backbone (SAM3 default
                1008). Encoder cost scales with its square, so this - not the view size - is the real
                speed/accuracy dial. Keep it a multiple of 112 to suit the patch grid.
        """
        # Check if SAM3 is available
        if not SAM3_AVAILABLE:
            raise ImportError(
                "SAM3 is not available. Please install SAM3:\n"
                "  1. Clone the repository: git clone https://github.com/facebookresearch/sam3.git\n"
                "  2. Install dependencies: cd sam3 && pip install -e .\n"
                "  3. Or set SAM3_PATH environment variable to the repository path"
            )
        
        self.device = device
        self.max_image_side = max_image_side if (max_image_side and max_image_side > 0) else None
        self.use_amp = bool(use_amp)
        self.clear_cache_each_image = bool(clear_cache_each_image)
        self.model_resolution = int(model_resolution)
        # Use provided token, or env var, or default token
        self.hf_token = hf_token or os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN') or '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'
        
        # Set Hugging Face token for authentication
        os.environ['HF_TOKEN'] = self.hf_token
        os.environ['HUGGINGFACE_TOKEN'] = self.hf_token
        
        # Login to Hugging Face if token is provided
        if self.hf_token:
            try:
                from huggingface_hub import login
                login(token=self.hf_token, add_to_git_credential=False)
            except Exception as e:
                print(f"Warning: Could not login to Hugging Face: {e}")
                print("Model download may fail if checkpoints require authentication")
        
        # Set PyTorch CUDA memory allocator config to reduce fragmentation
        if device == "cuda" and torch.cuda.is_available():
            if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                print("Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce memory fragmentation")
        
        # Check GPU memory if using CUDA
        if device == "cuda" and torch.cuda.is_available():
            print("Checking GPU memory availability...")
            has_memory, free_mem, total_mem, mem_msg = check_gpu_memory(min_gpu_memory_gb)
            print(mem_msg)
            
            # List other GPU processes
            if not has_memory:
                print("\n" + list_gpu_processes())
            
            if not has_memory:
                if auto_fallback_to_cpu:
                    print(f"\nAuto-fallback enabled: Switching to CPU due to insufficient GPU memory.")
                    print("Note: CPU inference will be significantly slower.")
                    device = "cpu"
                    self.device = "cpu"
                else:
                    error_msg = (
                        f"\nERROR: Insufficient GPU memory to load SAM3 model.\n"
                        f"Solutions:\n"
                        f"  1. Free up GPU memory by stopping other processes (check with: nvidia-smi)\n"
                        f"  2. Use CPU mode: Set device='cpu' or use --device cpu\n"
                        f"  3. Enable auto-fallback: Set auto_fallback_to_cpu=True\n"
                        f"  4. Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (already set)\n"
                        f"  5. Restart Python process to free GPU memory\n"
                    )
                    raise RuntimeError(error_msg)
        
        # Clear GPU cache before loading model (more aggressive)
        if device == "cuda" and torch.cuda.is_available():
            print("Clearing GPU cache before model loading...")
            # Multiple passes for better clearing
            for i in range(3):
                clear_gpu_memory()
            print("GPU cache cleared.")
        
        # Initialize SAM3 model with retry logic
        max_retries = 2
        for retry in range(max_retries):
            try:
                # build_sam3_image_model downloads from HF automatically when load_from_HF=True (default)
                self.model = build_sam3_image_model(
                    device=device,
                    eval_mode=True,
                    load_from_HF=True,
                    enable_segmentation=True
                )
                self.processor = Sam3Processor(
                    self.model, 
                    device=device,
                    confidence_threshold=confidence_threshold,
                    resolution=self.model_resolution
                )
                print(f"✓ SAM3 model loaded successfully on {device.upper()}")
                break  # Success, exit retry loop
            except RuntimeError as e:
                error_str = str(e)
                if "CUDA out of memory" in error_str or "out of memory" in error_str.lower():
                    print(f"\nCUDA Out of Memory Error while loading model (attempt {retry + 1}/{max_retries}):")
                    print(error_str)
                    
                    if device == "cuda" and not auto_fallback_to_cpu:
                        if retry < max_retries - 1:
                            print("\nTrying to recover...")
                            # More aggressive clearing
                            for _ in range(5):
                                clear_gpu_memory()
                            import time
                            time.sleep(2)  # Wait a bit for memory to be freed
                            print("Retrying model loading...")
                            continue
                        else:
                            print("\n" + list_gpu_processes())
                            print("\nSuggestions:")
                            print("  1. Stop other GPU processes (check with: nvidia-smi)")
                            print("  2. Use CPU mode: Set device='cpu' or use --device cpu")
                            print("  3. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (already set)")
                            print("  4. Restart Python process to free GPU memory")
                            print("  5. Kill other processes: kill -9 <PID>")
                    raise RuntimeError(f"CUDA OOM: {error_str}\nTry using CPU mode or freeing GPU memory.")
                else:
                    print(f"Error loading SAM3 model: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
            except Exception as e:
                print(f"Error loading SAM3 model: {e}")
                import traceback
                traceback.print_exc()
                raise
    
    def detect_and_segment(self, image_path, text_prompt,
                          box_threshold=0.32, text_threshold=0.32, prompt_mode="first_match"):
        """
        Perform detection and segmentation on an image
        
        Args:
            image_path: Path to input image
            text_prompt: Text prompt describing what to detect (e.g., "car . person . dog")
            box_threshold: Box threshold for filtering detections (not used by SAM3 directly)
            text_threshold: Text threshold for filtering detections (not used by SAM3 directly)
            prompt_mode: "first_match" to stop at first concept with detections, "all" to merge all concepts
        
        Returns:
            tuple: (annotated_image, masks, boxes, logits, phrases, processing_time)
        """
        start_time = time.time()
        
        # Load image with error handling for corrupted/truncated images
        # Try PIL first, fallback to OpenCV if PIL fails (OpenCV is more lenient with corrupted images)
        try:
            image = Image.open(image_path).convert("RGB")
            image_array = np.array(image)
        except (OSError, IOError, Image.UnidentifiedImageError) as e:
            # Fallback to OpenCV for corrupted/truncated images
            try:
                print(f"Warning: PIL failed to load '{image_path}', trying OpenCV fallback...")
                image_array = cv2.imread(image_path)
                if image_array is None:
                    raise ValueError(f"Failed to load image '{image_path}': Both PIL and OpenCV failed. The image may be corrupted or in an unsupported format.")
                # Convert BGR to RGB for consistency
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image_array)
            except Exception as e2:
                raise ValueError(f"Failed to load image '{image_path}': PIL error: {e}, OpenCV error: {e2}. The image may be corrupted or in an unsupported format.")

        original_h, original_w = image_array.shape[:2]

        # Resize image for inference (keeps GPU-only inference but avoids huge mask upsampling on GPU)
        resized = False
        scale_x = 1.0
        scale_y = 1.0
        model_image = image
        if self.max_image_side is not None:
            max_side = max(original_h, original_w)
            if max_side > self.max_image_side:
                resized = True
                scale = float(self.max_image_side) / float(max_side)
                new_w = max(1, int(round(original_w * scale)))
                new_h = max(1, int(round(original_h * scale)))
                # PIL resize expects (w, h)
                model_image = image.resize((new_w, new_h), resample=Image.BILINEAR)
                scale_x = float(original_w) / float(new_w)
                scale_y = float(original_h) / float(new_h)
        
        if self.processor is None:
            raise ValueError("SAM3 processor not initialized")
        
        use_cuda = self.device == "cuda" and torch.cuda.is_available()
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if (use_cuda and self.use_amp)
            else contextlib.nullcontext()
        )

        # Set image for SAM3 processor
        with autocast_ctx:
            state = self.processor.set_image(model_image)
        
        # Parse text prompt - handle multiple concepts separated by '.' or 'and'
        # Split by common separators and clean up
        text_prompt_clean = text_prompt.strip()
        concepts = []
        
        # Try splitting by various separators
        if ' . ' in text_prompt_clean or '.' in text_prompt_clean:
            concepts = [c.strip() for c in text_prompt_clean.replace(' . ', '.').split('.') if c.strip()]
        elif ' and ' in text_prompt_clean.lower():
            concepts = [c.strip() for c in text_prompt_clean.lower().split(' and ') if c.strip()]
        else:
            # Single concept
            concepts = [text_prompt_clean]
        
        # Remove empty concepts
        concepts = [c for c in concepts if c]
        
        if not concepts:
            concepts = [text_prompt_clean]
        
        # Collect all detections from all concepts
        all_boxes = []
        all_masks = []
        all_scores = []
        all_phrases = []
        
        # Process each concept separately and combine results
        # Keep a copy of the initial state with image features
        initial_state = state.copy()
        
        for i, concept in enumerate(concepts):
            # For subsequent concepts, reset prompts but keep backbone_out (image features)
            if i > 0:
                # Reset prompts but keep the image backbone output
                state = initial_state.copy()
            
            # Set text prompt for this concept
            with autocast_ctx:
                state = self.processor.set_text_prompt(prompt=concept, state=state)
            
            # Extract results
            masks = state.get("masks", None)
            boxes = state.get("boxes", None)
            scores = state.get("scores", None)
            
            # Convert to numpy arrays if they are tensors
            if masks is not None:
                if isinstance(masks, torch.Tensor):
                    masks = masks.cpu().numpy()
                masks = np.array(masks)
            else:
                masks = np.array([])
                
            if boxes is not None:
                if isinstance(boxes, torch.Tensor):
                    boxes = boxes.cpu().numpy()
                boxes = np.array(boxes)
            else:
                boxes = np.array([])
                
            if scores is not None:
                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy()
                scores = np.array(scores)
            else:
                scores = np.array([])
            
            # Add to combined results
            if len(boxes) > 0:
                all_boxes.append(boxes)
                all_masks.append(masks if len(masks) > 0 else np.array([]))
                all_scores.append(scores)
                all_phrases.extend([concept] * len(boxes))
                if prompt_mode == "first_match":
                    break

            # Free prompt-specific tensors ASAP (helps long runs / multiple concepts)
            try:
                for k in ("masks", "masks_logits", "boxes", "scores"):
                    if k in state:
                        del state[k]
            except Exception:
                pass
            if use_cuda and self.clear_cache_each_image:
                clear_gpu_memory()
        
        # Combine all results
        if len(all_boxes) > 0:
            boxes = np.concatenate(all_boxes, axis=0)
            scores = np.concatenate(all_scores, axis=0)
            phrases = all_phrases
            
            # Combine masks
            mask_list = [m for m in all_masks if len(m) > 0]
            if len(mask_list) > 0:
                masks = np.concatenate(mask_list, axis=0)
            else:
                masks = np.array([])
        else:
            boxes = np.array([])
            masks = np.array([])
            scores = np.array([])
            phrases = []
        
        # SAM3 already filters by confidence_threshold, but we can apply additional filtering
        # Filter by threshold if boxes and scores are available
        if len(boxes) > 0 and len(scores) > 0:
            # Apply additional threshold filtering if needed
            valid_indices = scores >= box_threshold
            
            boxes = boxes[valid_indices]
            scores = scores[valid_indices]
            
            if len(masks) > 0:
                masks = masks[valid_indices]
            
            phrases = [phrases[i] for i in range(len(phrases)) if (i < len(valid_indices) and valid_indices[i])]
        
        # Ensure boxes are in correct format (SAM3 returns xyxy format, already scaled)
        if len(boxes) > 0:
            input_boxes = boxes.reshape(-1, 4) if boxes.ndim > 1 else boxes
        else:
            input_boxes = np.array([]).reshape(0, 4)

        # If we resized for inference, scale boxes back to original image coordinates
        if resized and len(input_boxes) > 0:
            input_boxes = input_boxes.astype(np.float32, copy=False)
            input_boxes[:, [0, 2]] *= scale_x
            input_boxes[:, [1, 3]] *= scale_y
        
        # Ensure masks are in correct format (SAM3 returns boolean masks)
        if len(masks) > 0:
            if masks.ndim == 3:
                # Shape: (n, h, w) - already correct
                masks = masks
            elif masks.ndim == 2:
                # Single mask, add batch dimension
                masks = masks[np.newaxis, :, :]
            elif masks.ndim == 4:
                # Remove extra dimension if present
                masks = masks.squeeze(1)
        else:
            masks = np.array([])

        # If we resized for inference, upscale masks back to original image size on CPU (avoids GPU OOM)
        if resized and len(masks) > 0:
            resized_h, resized_w = masks.shape[-2], masks.shape[-1]
            if (resized_h, resized_w) != (original_h, original_w):
                upscaled = []
                for m in masks:
                    m_uint8 = (m.astype(np.uint8) if m.dtype == bool else (m > 0.5).astype(np.uint8))
                    m_up = cv2.resize(
                        m_uint8,
                        (original_w, original_h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                    upscaled.append(m_up)
                masks = np.stack(upscaled, axis=0) if upscaled else np.array([])
        
        # Use scores as logits/confidences
        logits = scores if len(scores) > 0 else np.array([])
        
        # Annotate image with masks only (no bounding boxes, no labels)
        # Use INDEX color lookup since we don't have class_id in detections
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        
        # Create detections object for masks only
        if len(input_boxes) > 0:
            detections = sv.Detections(
                xyxy=input_boxes,
                mask=masks.astype(bool) if len(masks) > 0 else None
            )
        else:
            detections = sv.Detections.empty()
        
        annotated_frame = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        if len(input_boxes) > 0:
            # Draw masks only (no bounding boxes, no labels)
            if len(masks) > 0:
                annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        
        # Convert back to RGB for consistency
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # Calculate processing time
        processing_time = time.time() - start_time

        # Final cleanup for this image
        try:
            del state
            del initial_state
        except Exception:
            pass
        if use_cuda and self.clear_cache_each_image:
            clear_gpu_memory()
        
        return annotated_frame, masks, input_boxes, logits, phrases, processing_time

    def detect_and_segment_from_image(self, image, text_prompt,
                                       box_threshold=0.32, text_threshold=0.32, prompt_mode="first_match"):
        """
        Same as detect_and_segment but accepts image in memory (PIL Image or numpy array RGB).
        Used for auto-label only (no file path).
        """
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_array = image
                image = Image.fromarray(image_array.astype(np.uint8))
            else:
                image_array = image
                image = Image.fromarray(image_array.astype(np.uint8))
        else:
            image_array = np.array(image)
        original_h, original_w = image_array.shape[:2]

        resized = False
        scale_x = 1.0
        scale_y = 1.0
        model_image = image
        if self.max_image_side is not None:
            max_side = max(original_h, original_w)
            if max_side > self.max_image_side:
                resized = True
                scale = float(self.max_image_side) / float(max_side)
                new_w = max(1, int(round(original_w * scale)))
                new_h = max(1, int(round(original_h * scale)))
                model_image = image.resize((new_w, new_h), resample=Image.BILINEAR)
                scale_x = float(original_w) / float(new_w)
                scale_y = float(original_h) / float(new_h)

        if self.processor is None:
            raise ValueError("SAM3 processor not initialized")

        use_cuda = self.device == "cuda" and torch.cuda.is_available()
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if (use_cuda and self.use_amp)
            else contextlib.nullcontext()
        )

        with autocast_ctx:
            state = self.processor.set_image(model_image)

        text_prompt_clean = text_prompt.strip()
        concepts = []
        if ' . ' in text_prompt_clean or '.' in text_prompt_clean:
            concepts = [c.strip() for c in text_prompt_clean.replace(' . ', '.').split('.') if c.strip()]
        elif ' and ' in text_prompt_clean.lower():
            concepts = [c.strip() for c in text_prompt_clean.lower().split(' and ') if c.strip()]
        else:
            concepts = [text_prompt_clean]
        concepts = [c for c in concepts if c]
        if not concepts:
            concepts = [text_prompt_clean]

        all_boxes = []
        all_masks = []
        all_scores = []
        all_phrases = []
        initial_state = state.copy()

        for i, concept in enumerate(concepts):
            if i > 0:
                state = initial_state.copy()
            with autocast_ctx:
                state = self.processor.set_text_prompt(prompt=concept, state=state)
            masks = state.get("masks", None)
            boxes = state.get("boxes", None)
            scores = state.get("scores", None)
            if masks is not None:
                if isinstance(masks, torch.Tensor):
                    masks = masks.cpu().numpy()
                masks = np.array(masks)
            else:
                masks = np.array([])
            if boxes is not None:
                if isinstance(boxes, torch.Tensor):
                    boxes = boxes.cpu().numpy()
                boxes = np.array(boxes)
            else:
                boxes = np.array([])
            if scores is not None:
                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy()
                scores = np.array(scores)
            else:
                scores = np.array([])
            if len(boxes) > 0:
                all_boxes.append(boxes)
                all_masks.append(masks if len(masks) > 0 else np.array([]))
                all_scores.append(scores)
                all_phrases.extend([concept] * len(boxes))
                if prompt_mode == "first_match":
                    break
            try:
                for k in ("masks", "masks_logits", "boxes", "scores"):
                    if k in state:
                        del state[k]
            except Exception:
                pass
            if use_cuda and self.clear_cache_each_image:
                clear_gpu_memory()

        if len(all_boxes) > 0:
            boxes = np.concatenate(all_boxes, axis=0)
            scores = np.concatenate(all_scores, axis=0)
            phrases = all_phrases
            mask_list = [m for m in all_masks if len(m) > 0]
            masks = np.concatenate(mask_list, axis=0) if mask_list else np.array([])
        else:
            boxes = np.array([])
            masks = np.array([])
            scores = np.array([])
            phrases = []

        if len(boxes) > 0 and len(scores) > 0:
            valid_indices = scores >= box_threshold
            boxes = boxes[valid_indices]
            scores = scores[valid_indices]
            if len(masks) > 0:
                masks = masks[valid_indices]
            phrases = [phrases[i] for i in range(len(phrases)) if (i < len(valid_indices) and valid_indices[i])]

        input_boxes = boxes.reshape(-1, 4) if len(boxes) > 0 else np.array([]).reshape(0, 4)
        if resized and len(input_boxes) > 0:
            input_boxes = input_boxes.astype(np.float32, copy=False)
            input_boxes[:, [0, 2]] *= scale_x
            input_boxes[:, [1, 3]] *= scale_y

        if len(masks) > 0 and masks.ndim == 2:
            masks = masks[np.newaxis, :, :]
        logits = scores if len(scores) > 0 else np.array([])
        return None, masks, input_boxes, logits, phrases, 0.0


class S3ImageProcessor:
    """Handle S3 image downloads and processing"""
    
    def __init__(self, aws_access_key_id=None, aws_secret_access_key=None, 
                 aws_region='us-east-1'):
        """
        Initialize S3 client
        
        Args:
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
            aws_region: AWS region
        """
        try:
            if aws_access_key_id and aws_secret_access_key:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=aws_region
                )
            else:
                # Use default credentials from environment or IAM role
                self.s3_client = boto3.client('s3', region_name=aws_region)
        except Exception as e:
            print(f"Error initializing S3 client: {e}")
            self.s3_client = None
    
    def download_image(self, s3_path, local_path):
        """
        Download image from S3
        
        Args:
            s3_path: S3 path in format 's3://bucket-name/key' or 'bucket-name/key'
            local_path: Local path to save the image
        
        Returns:
            str: Path to downloaded image
        """
        if self.s3_client is None:
            raise ValueError("S3 client not initialized")
        
        # Parse S3 path
        if s3_path.startswith('s3://'):
            s3_path = s3_path[5:]
        
        parts = s3_path.split('/', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 path format: {s3_path}")
        
        bucket_name, key = parts
        
        # Create local directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download file silently
        self.s3_client.download_file(bucket_name, key, local_path)
        
        return local_path
    
    def upload_image(self, local_path, s3_path):
        """
        Upload image to S3
        
        Args:
            local_path: Local path to the image
            s3_path: S3 path in format 's3://bucket-name/key' or 'bucket-name/key'
        """
        if self.s3_client is None:
            raise ValueError("S3 client not initialized")
        
        # Parse S3 path
        if s3_path.startswith('s3://'):
            s3_path = s3_path[5:]
        
        parts = s3_path.split('/', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 path format: {s3_path}")
        
        bucket_name, key = parts
        
        # Upload file silently
        self.s3_client.upload_file(local_path, bucket_name, key)


def save_segmentation_results(image, masks, boxes, phrases, output_path, image_path=None, scores=None, combined=False):
    """
    Save segmentation results as separate overlay images for each instance.
    Each instance gets its own overlay file labeled as 'object 1', 'object 2', etc.
    
    Args:
        image: Annotated image (RGB numpy array) - used as fallback if image_path not provided
        masks: numpy array of masks (shape: [N, H, W] where N is number of masks)
        boxes: numpy array of bounding boxes (shape: [N, 4])
        phrases: List of phrases/labels for each detection
        output_path: Base output path (will create object_1, object_2, etc. files)
        image_path: Optional path to original image (preferred over annotated image)
        scores: Optional confidence scores aligned with detections
    """
    import supervision as sv
    
    # Load original image - prefer image_path if provided, otherwise use the annotated image
    if image_path:
        # Load original image from file
        try:
            original_image_pil = Image.open(image_path).convert("RGB")
            original_image = np.array(original_image_pil)
        except (OSError, IOError, Image.UnidentifiedImageError) as e:
            # Fallback to OpenCV
            try:
                print(f"Warning: PIL failed to load '{image_path}' in save_segmentation_results, trying OpenCV fallback...")
                original_image = cv2.imread(image_path)
                if original_image is None:
                    raise ValueError(f"Failed to load image '{image_path}'")
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            except Exception as e2:
                print(f"Warning: Could not load image from {image_path}, using provided image. Error: {e2}")
                # Fallback to provided image
                if isinstance(image, np.ndarray):
                    original_image = image.copy()
                    if len(original_image.shape) == 3 and original_image.shape[2] == 3:
                        pass  # Already RGB
                    else:
                        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                else:
                    original_image = np.array(image)
                    if len(original_image.shape) == 3 and original_image.shape[2] == 3:
                        pass  # Already RGB
                    else:
                        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    else:
        # Use provided image (may already be annotated, but we'll work with it)
        if isinstance(image, np.ndarray):
            original_image = image.copy()
            if len(original_image.shape) == 3 and original_image.shape[2] == 3:
                pass  # Already RGB
            else:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        else:
            original_image = np.array(image)
            if len(original_image.shape) == 3 and original_image.shape[2] == 3:
                pass  # Already RGB
            else:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Convert to BGR for OpenCV operations
    original_image_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    
    # Get base path and extension
    base_path = os.path.splitext(output_path)[0]
    ext = os.path.splitext(output_path)[1] or '.jpg'
    
    saved_paths = []
    
    # If no masks, save a single empty overlay
    if len(masks) == 0:
        cv2.imwrite(output_path, original_image_bgr)
        saved_paths.append(output_path)
        return saved_paths
    
    # Create separate overlay for each instance
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)

    if combined:
        # Single overlay containing every detection for this image
        all_masks = masks.astype(bool) if masks.ndim == 3 else masks[np.newaxis, :, :].astype(bool)
        all_boxes = boxes if len(boxes) > 0 else np.array([]).reshape(0, 4)

        overlay_image = original_image_bgr.copy()
        if len(all_boxes) > 0:
            detections = sv.Detections(xyxy=all_boxes, mask=all_masks)
            overlay_image = mask_annotator.annotate(scene=overlay_image, detections=detections)
            overlay_image = box_annotator.annotate(scene=overlay_image, detections=detections)

            labels = []
            for idx in range(len(all_boxes)):
                label = f"object {idx + 1}"
                if idx < len(phrases):
                    label += f" ({phrases[idx]})"
                if scores is not None and idx < len(scores):
                    label += f" score={float(scores[idx]):.2f}"
                labels.append(label)
            overlay_image = label_annotator.annotate(scene=overlay_image, detections=detections, labels=labels)

        cv2.imwrite(output_path, overlay_image)
        return [output_path]
    
    for idx in range(len(masks)):
        # Create a copy of the original image for this instance
        overlay_image = original_image_bgr.copy()
        
        # Get single mask and box for this instance
        single_mask = masks[idx:idx+1] if masks.ndim == 3 else masks[idx]
        if single_mask.ndim == 2:
            single_mask = single_mask[np.newaxis, :, :]
        
        single_box = boxes[idx:idx+1] if len(boxes) > 0 else np.array([]).reshape(0, 4)
        
        # Create detections object for this single instance
        if len(single_box) > 0:
            detections = sv.Detections(
                xyxy=single_box,
                mask=single_mask.astype(bool) if len(single_mask) > 0 else None
            )
        else:
            detections = sv.Detections.empty()
        
        # Draw mask overlay
        if len(single_mask) > 0:
            overlay_image = mask_annotator.annotate(scene=overlay_image, detections=detections)
        
        # Draw bounding box
        if len(single_box) > 0:
            overlay_image = box_annotator.annotate(scene=overlay_image, detections=detections)
        
        # Add label "object N"
        object_label = f"object {idx + 1}"
        if idx < len(phrases):
            object_label += f" ({phrases[idx]})"
        if scores is not None and idx < len(scores):
            object_label += f" score={float(scores[idx]):.2f}"
        
        if len(single_box) > 0:
            labels = [object_label]
            overlay_image = label_annotator.annotate(scene=overlay_image, detections=detections, labels=labels)
        
        # Save individual overlay file
        object_output_path = f"{base_path}_object_{idx + 1}{ext}"
        cv2.imwrite(object_output_path, overlay_image)
        saved_paths.append(object_output_path)
    
    return saved_paths


def save_masked_images(image_path, masks, output_path):
    """
    Save binary masked images where masked regions are in color and rest is black.
    
    Args:
        image_path: Path to the original input image
        masks: numpy array of masks (shape: [N, H, W] where N is number of masks)
        output_path: Path to save the masked output image
    """
    # Load original image with error handling for corrupted/truncated images
    # Try PIL first, fallback to OpenCV if PIL fails
    try:
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
    except (OSError, IOError, Image.UnidentifiedImageError) as e:
        # Fallback to OpenCV for corrupted/truncated images
        try:
            print(f"Warning: PIL failed to load '{image_path}' in save_masked_images, trying OpenCV fallback...")
            image_array = cv2.imread(image_path)
            if image_array is None:
                raise ValueError(f"Failed to load image '{image_path}': Both PIL and OpenCV failed. The image may be corrupted.")
            # Convert BGR to RGB for consistency
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        except Exception as e2:
            raise ValueError(f"Failed to load image '{image_path}': PIL error: {e}, OpenCV error: {e2}. The image may be corrupted.")
    
    # Create combined mask from all masks
    if len(masks) == 0:
        # No masks, return black image
        masked_image = np.zeros_like(image_array)
    else:
        # Combine all masks (OR operation)
        combined_mask = np.zeros(masks[0].shape, dtype=bool)
        for mask in masks:
            if isinstance(mask, np.ndarray):
                # Ensure mask is boolean
                if mask.dtype != bool:
                    mask = mask > 0.5
                combined_mask = combined_mask | mask
        
        # Apply mask: keep masked regions in color, make rest black
        masked_image = image_array.copy()
        # Set non-masked regions to black
        masked_image[~combined_mask] = [0, 0, 0]
    
    # Convert RGB to BGR for OpenCV and save
    masked_image_bgr = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, masked_image_bgr)


def save_binary_masks(masks, output_dir, image_name=None, image_shape=None, combined=False):
    """
    Save separate binary mask images for each instance (same pattern as overlays and colored segments).
    Each instance gets its own binary mask file: white (255) for segmented region, black (0) elsewhere.
    Files are named e.g. output_dir/<image_basename>_object_1.png, _object_2.png, etc.

    Args:
        masks: numpy array of masks (shape: [N, H, W] where N is number of masks)
        output_dir: Directory to save binary mask images
        image_name: Optional image name for filenames (e.g. "frame.jpg" -> frame_object_1.png, frame_object_2.png)
        image_shape: Optional tuple (height, width) for empty masks or resize; inferred from first mask if not given

    Returns:
        list: Paths to saved binary mask files
    """
    os.makedirs(output_dir, exist_ok=True)

    if image_name:
        image_basename = os.path.splitext(os.path.basename(image_name))[0]
    else:
        image_basename = "image"

    saved_paths = []

    if len(masks) == 0:
        return saved_paths

    if combined:
        # Union of every instance mask in a single binary image
        merged = None
        for mask in masks:
            if not isinstance(mask, np.ndarray):
                continue
            mask_bool = mask > 0.5 if mask.dtype != bool else mask
            if mask_bool.ndim > 2:
                mask_bool = mask_bool.squeeze()
            if mask_bool.ndim != 2:
                continue
            merged = mask_bool if merged is None else (merged | mask_bool)

        if merged is None:
            return saved_paths

        mask_path = os.path.join(output_dir, f"{image_basename}.png")
        Image.fromarray(merged.astype(np.uint8) * 255, mode="L").save(mask_path)
        return [mask_path]

    for idx, mask in enumerate(masks):
        if not isinstance(mask, np.ndarray):
            continue

        if mask.dtype != bool:
            mask_bool = mask > 0.5
        else:
            mask_bool = mask

        if mask_bool.ndim > 2:
            mask_bool = mask_bool.squeeze()
        if mask_bool.ndim != 2:
            continue

        binary_mask = mask_bool.astype(np.uint8) * 255
        mask_path = os.path.join(output_dir, f"{image_basename}_object_{idx + 1}.png")
        mask_image = Image.fromarray(binary_mask, mode="L")
        mask_image.save(mask_path)
        saved_paths.append(mask_path)

    return saved_paths


def save_colored_segmented_images(image_path, masks, output_dir, image_name=None, combined=False):
    """
    Save separate colored segmented images for each instance.
    Each instance gets its own colored segment file labeled as 'object 1', 'object 2', etc.
    
    Args:
        image_path: Path to the original input image
        masks: numpy array of masks (shape: [N, H, W] where N is number of masks)
        output_dir: Directory to save colored segmented images
        image_name: Optional image name to use in filename (e.g., colored_segments/<image_name>_object_1.png)
    
    Returns:
        list: List of paths to saved colored segmented images
    """
    # Color palette for different segmented regions (RGB format)
    # Similar to vinay.py colors: Red, Green, Blue, Yellow, Magenta, Cyan, etc.
    color_palette = [
        [255, 0, 0],      # Red
        [0, 255, 0],      # Green
        [0, 0, 255],      # Blue
        [255, 255, 0],    # Yellow
        [255, 0, 255],    # Magenta
        [0, 255, 255],    # Cyan
        [128, 0, 128],    # Purple
        [255, 165, 0],    # Orange
        [0, 128, 255],    # Light Blue
        [128, 255, 0],    # Lime
        [255, 192, 203],  # Pink
        [165, 42, 42],    # Brown
        [128, 128, 128],  # Gray
        [255, 20, 147],   # Deep Pink
        [0, 191, 255],    # Deep Sky Blue
        [205, 133, 63],   # Peru (periwinkle-like)
    ]
    
    # Create output directory (no subdirectories)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image basename for filename if provided
    if image_name:
        image_basename = os.path.splitext(os.path.basename(image_name))[0]
    else:
        image_basename = os.path.splitext(os.path.basename(image_path))[0]
    
    saved_paths = []
    
    # Load original image to get dimensions (needed even for blank images)
    # Try PIL first, fallback to OpenCV if PIL fails (OpenCV is more lenient with corrupted images)
    try:
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
    except (OSError, IOError, Image.UnidentifiedImageError) as e:
        # Fallback to OpenCV for corrupted/truncated images
        try:
            print(f"Warning: PIL failed to load '{image_path}' in save_colored_segmented_images, trying OpenCV fallback...")
            image_array = cv2.imread(image_path)
            if image_array is None:
                print(f"Error loading image {image_path}: Both PIL and OpenCV failed. The image may be corrupted.")
                return saved_paths
            # Convert BGR to RGB for consistency
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        except Exception as e2:
            print(f"Error loading image {image_path}: PIL error: {e}, OpenCV error: {e2}. The image may be corrupted.")
            return saved_paths
    
    # If no masks, return empty list
    if len(masks) == 0:
        print(f"No masks detected for {image_basename}, skipping colored segment generation")
        return saved_paths
    
    if combined:
        # All instances painted into one image, each in its own palette color
        colored_image = np.zeros_like(image_array)
        valid_masks = 0
        for idx, mask in enumerate(masks):
            if not isinstance(mask, np.ndarray):
                continue
            mask_bool = mask > 0.5 if mask.dtype != bool else mask
            if mask_bool.ndim > 2:
                mask_bool = mask_bool.squeeze()
            if mask_bool.ndim != 2:
                continue
            if mask_bool.shape != image_array.shape[:2]:
                mask_bool = cv2.resize(
                    mask_bool.astype(np.uint8),
                    (image_array.shape[1], image_array.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ) > 0.5
            colored_image[mask_bool] = color_palette[idx % len(color_palette)]
            valid_masks += 1

        if valid_masks == 0:
            return saved_paths

        mask_path = os.path.join(output_dir, f"{image_basename}.png")
        cv2.imwrite(mask_path, cv2.cvtColor(colored_image, cv2.COLOR_RGB2BGR))
        return [mask_path]

    # Process each mask separately and save individual colored segment files
    valid_masks = 0
    for idx, mask in enumerate(masks):
        if not isinstance(mask, np.ndarray):
            continue
        
        # Ensure mask is boolean
        if mask.dtype != bool:
            mask_bool = mask > 0.5
        else:
            mask_bool = mask
        
        # Ensure mask is 2D
        if mask_bool.ndim > 2:
            mask_bool = mask_bool.squeeze()
        
        if mask_bool.ndim != 2:
            print(f"Warning: Skipping mask {idx} - invalid shape: {mask_bool.shape}")
            continue
        
        # Resize mask to match image size if needed
        if mask_bool.shape != image_array.shape[:2]:
            mask_bool = cv2.resize(
                mask_bool.astype(np.uint8), 
                (image_array.shape[1], image_array.shape[0]), 
                interpolation=cv2.INTER_NEAREST
            ) > 0.5
        
        # Get color for this mask (cycle through palette)
        color = color_palette[idx % len(color_palette)]
        
        # Create colored image for this single instance: start with black background
        colored_image = np.zeros_like(image_array)
        
        # Apply color to masked region
        colored_image[mask_bool] = color
        
        # Save individual colored segmented image
        colored_image_bgr = cv2.cvtColor(colored_image, cv2.COLOR_RGB2BGR)
        mask_path = os.path.join(output_dir, f"{image_basename}_object_{idx + 1}.png")
        cv2.imwrite(mask_path, colored_image_bgr)
        saved_paths.append(mask_path)
        valid_masks += 1
    
    if valid_masks == 0:
        print(f"No valid masks saved for {image_basename}")
    else:
        print(f"Saved {valid_masks} individual colored segmented images for {image_basename}")
    
    return saved_paths


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 Detection and Segmentation for S3 Images"
    )
    
    # S3 configuration
    parser.add_argument(
        "--s3_image_path",
        type=str,
        required=True,
        help="S3 path to input image (e.g., 's3://bucket-name/path/to/image.jpg')"
    )
    parser.add_argument(
        "--text_prompt",
        type=str,
        required=True,
        help="Text prompt for detection (e.g., 'car . person . dog')"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save output images"
    )
    parser.add_argument(
        "--colored_segments_dir",
        type=str,
        default=None,
        help="Directory to save colored segmented image with all masks combined (format: colored_segments/<image_name>.png). If not specified, uses output_dir/colored_segments"
    )
    parser.add_argument(
        "--output_s3_path",
        type=str,
        default=None,
        help="Optional S3 path to upload results (e.g., 's3://bucket-name/output.jpg')"
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
    
    # AWS credentials (optional, can use environment variables or IAM role)
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
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create colored segments output directory
    if args.colored_segments_dir is None:
        colored_segments_dir = os.path.join(args.output_dir, "colored_segments")
    else:
        colored_segments_dir = args.colored_segments_dir
    os.makedirs(colored_segments_dir, exist_ok=True)
    
    # Initialize S3 processor
    s3_processor = S3ImageProcessor(
        aws_access_key_id=args.aws_access_key_id,
        aws_secret_access_key=args.aws_secret_access_key,
        aws_region=args.aws_region
    )
    
    # Download image from S3 to temporary location
    import tempfile
    temp_dir = tempfile.mkdtemp()
    input_filename = os.path.basename(args.s3_image_path)
    local_input_path = os.path.join(temp_dir, input_filename)
    
    try:
        s3_processor.download_image(args.s3_image_path, local_input_path)
    except Exception as e:
        print(f"Error downloading image from S3: {e}")
        return
    
    # Use provided token, env var, or default
    hf_token = args.hf_token or os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN') or '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'
    
    # Initialize SAM3 detector
    try:
        detector = SAM3Detector(
            hf_token=hf_token,
            confidence_threshold=args.confidence_threshold
        )
    except Exception as e:
        print(f"Error initializing SAM3 model: {e}")
        print("\nPlease ensure:")
        print("1. SAM3 repository is cloned: git clone https://github.com/facebookresearch/sam3.git")
        print("2. SAM3 dependencies are installed: cd sam3 && pip install -e .")
        print("3. Hugging Face token is set (HF_TOKEN env var or --hf_token argument)")
        import traceback
        traceback.print_exc()
        return
    
    # Perform detection and segmentation
    try:
        annotated_image, masks, boxes, logits, phrases, processing_time = detector.detect_and_segment(
            image_path=local_input_path,
            text_prompt=args.text_prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold
        )
        
        # Save results (use original filename without "input_" prefix)
        output_filename = input_filename
        output_path = os.path.join(args.output_dir, output_filename)
        overlay_paths = save_segmentation_results(
            annotated_image,
            masks,
            boxes,
            phrases,
            output_path,
            image_path=local_input_path,
            scores=logits,
        )
        
        # Save colored segmented images per region (like vinay.py format)
        save_colored_segmented_images(local_input_path, masks, colored_segments_dir, image_name=output_filename)
        
        # Clean up temporary input file
        try:
            os.remove(local_input_path)
            os.rmdir(temp_dir)
        except:
            pass
        
        # Upload to S3 if specified
        if args.output_s3_path:
            try:
                s3_processor.upload_image(output_path, args.output_s3_path)
            except Exception as e:
                print(f"Error uploading to S3: {e}")
        
    except Exception as e:
        print(f"Error during detection/segmentation: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

