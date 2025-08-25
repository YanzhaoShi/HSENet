import os
import numpy as np
import torch
from PIL import Image
import tqdm
import glob
import argparse
import multiprocessing
from multiprocessing import Pool
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor
from open_clip import create_model_from_pretrained

'''
The function of this script is to extract BiomedCLIP features from 2D CT slices and save them as npy files.
This script can process a single case or the entire dataset using multiple GPUs.
You can specify the number of GPUs and tasks per GPU.
Steps:
1. Set the 'base_dir' to the directory containing your CT slice directories.
2. Set the 'output_dir' to the directory where you want to save the npy files.
3. Optionally, set 'num_gpus' and 'tasks_per_gpu' to control parallel processing.
4. Optionally, set 'single_case' to process only one specific case.
5. Run the script, it will automatically extract features and save them in the specified output directory
'''

""" 
Input directory structure:
base_dir/
  └── case_1/
       └── slices/
            ├── slice_000.jpg
            ├── slice_001.jpg
            └── ...
  └── case_2/
       └── slices/
            ├── slice_000.jpg
            ├── slice_001.jpg
            └── ...
  
Output directory structure:
output_dir/
  └── case_1/
       └── case_1_biomedclip_features.npy
  └── case_2/
       └── case_2_biomedclip_features.npy
"""

def extract_features(slice_dir, output_npy, model, preprocess, device):
    """
    Extract BiomedCLIP features from slice directory and save as npy file
    
    Args:
        slice_dir: Directory containing JPG slices
        output_npy: Output npy file path
        model: Pre-loaded BiomedCLIP model
        preprocess: Pre-loaded preprocessing function
        device: Computing device
    """
    # Check output directory
    os.makedirs(os.path.dirname(output_npy), exist_ok=True)
    
    # Get all slice paths and sort
    slice_paths = sorted(glob.glob(os.path.join(slice_dir, "slice_*.jpg")))
    
    if not slice_paths:
        raise ValueError(f"No slice images found in {slice_dir}")
    
    print(f"Found {len(slice_paths)} slice images in {slice_dir}")
    
    # Load all images at once
    batch_images = []
    for path in slice_paths:
        try:
            img = Image.open(path).convert("RGB")
            batch_images.append(preprocess(img))
        except Exception as e:
            print(f"Error processing image {path}: {e}")
            # Use zero tensor to replace error image
            batch_images.append(torch.zeros((3, 224, 224)))
    
    # Process all images as a large batch
    batch_tensor = torch.stack(batch_images).to(device)
    
    # Extract features
    with torch.no_grad():
        # image_features = model.encode_image(batch_tensor)
        image_features = model.visual.trunk(batch_tensor)
    
    # Move features to CPU and convert to NumPy
    features_array = image_features.cpu().numpy()
    
    # Save features
    np.save(output_npy, features_array)
    print(f"Features saved to: {output_npy}")
    print(f"Feature shape: {features_array.shape}")
    
    return output_npy

def process_single_dir(args):
    """Function to process a single directory"""
    slice_dir, base_dir, output_base_dir, model, preprocess, device, task_id, gpu_id = args
    
    try:
        # Get relevant paths
        parent_dir = os.path.dirname(slice_dir)
        case_id = os.path.basename(parent_dir)
        
        # Calculate relative path, maintain the same structure in the new output directory
        rel_path = os.path.relpath(parent_dir, base_dir)
        new_parent_dir = os.path.join(output_base_dir, rel_path)
        os.makedirs(new_parent_dir, exist_ok=True)
        
        output_npy = os.path.join(new_parent_dir, f"{case_id}_biomedclip_features.npy")
        
        # Skip if already processed
        if os.path.exists(output_npy):
            print(f"[GPU {gpu_id}, Task {task_id}] Skipping already processed: {case_id}")
            return True
        
        # Extract features - pass in the already loaded model and preprocessing function
        extract_features(slice_dir, output_npy, model, preprocess, device)
        return True
    
    except Exception as e:
        print(f"[GPU {gpu_id}, Task {task_id}] Error when processing {slice_dir}: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_gpu_batch(batch_args):
    """Process a batch of directories assigned to a specific GPU, using thread pool for parallel tasks on GPU"""
    gpu_id, dir_batch, base_dir, output_base_dir, tasks_per_gpu = batch_args
    print(f"[GPU {gpu_id}] Starting to process {len(dir_batch)} directories with {tasks_per_gpu} parallel tasks")
    
    # Set GPU device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
    
    # Load model only once per GPU
    print(f"[GPU {gpu_id}] Loading BiomedCLIP model...")
    model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    model = model.to(device)
    model.eval()
    
    # Limit number of tasks per GPU
    num_threads = min(tasks_per_gpu, len(dir_batch))
    results = []
    
    # Use thread pool to process directories in parallel
    task_args = [(slice_dir, base_dir, output_base_dir, model, preprocess, device, i % num_threads, gpu_id) 
                for i, slice_dir in enumerate(dir_batch)]
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        futures = [executor.submit(process_single_dir, arg) for arg in task_args]
        
        # Use tqdm to display progress
        for future in tqdm.tqdm(
            futures, 
            desc=f"GPU {gpu_id} Processing", 
            position=gpu_id,
            total=len(dir_batch)
        ):
            results.append(future.result())
            
            # Periodically clear GPU cache
            if len(results) % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    success_count = sum(1 for r in results if r)
    failed_count = len(results) - success_count
    print(f"[GPU {gpu_id}] Processing complete: {success_count} successful, {failed_count} failed")
    
    # Clean up model after processing is complete
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results

def get_slice_dirs(base_dir):
    """Find slice directories using caching mechanism"""
    cache_file = os.path.join(os.path.dirname(base_dir), "slice_dirs_cache.txt")
    
    # Check if cache file exists and is fresh
    if os.path.exists(cache_file):
        cache_mtime = os.path.getmtime(cache_file)
        base_dir_mtime = os.path.getmtime(base_dir)
        
        # If cache is newer than base directory, use the cache
        if cache_mtime > base_dir_mtime:
            try:
                with open(cache_file, 'r') as f:
                    slice_dirs = [line.strip() for line in f if line.strip()]
                    if slice_dirs:
                        print(f"Loaded {len(slice_dirs)} directories from cache")
                        return slice_dirs
            except Exception as e:
                print(f"Cache read failed: {e}, re-searching")
    
    print("Searching directories, this may take a few minutes...")
    
    # Use system command for fast search
    try:
        cmd = f"find {base_dir} -type d -name 'slices'"
        result = subprocess.check_output(cmd, shell=True, text=True)
        slice_dirs = [line.strip() for line in result.split('\n') if line.strip()]
    except:
        # Fall back to Python search
        slice_dirs = []
        for root, dirs, files in os.walk(base_dir):
            if "slices" in dirs:
                slice_dirs.append(os.path.join(root, "slices"))
                # Print progress every 1000 directories
                if len(slice_dirs) % 1000 == 0:
                    print(f"Found {len(slice_dirs)} directories...")
    
    # Save to cache
    try:
        with open(cache_file, 'w') as f:
            for dir_path in slice_dirs:
                f.write(f"{dir_path}\n")
        print(f"Directory list cached to: {cache_file}")
    except Exception as e:
        print(f"Cache save failed: {e}")
    
    return slice_dirs

def process_dataset_with_multi_gpus(base_dir, output_base_dir, num_gpus=8, tasks_per_gpu=4):
    """
    Process the entire dataset using multiple GPUs, running multiple tasks on each GPU
    
    Args:
        base_dir: Dataset base directory
        output_base_dir: Feature output base directory
        num_gpus: Number of GPUs to use
        tasks_per_gpu: Number of parallel tasks per GPU
    """
    # Ensure output directory exists
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Get number of available GPUs
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    if available_gpus == 0:
        print("No GPUs available, will use CPU")
        num_gpus = 1
    elif available_gpus < num_gpus:
        print(f"Warning: Requested {num_gpus} GPUs, but only {available_gpus} detected")
        num_gpus = available_gpus
    
    print(f"Will use {num_gpus} {'GPUs' if num_gpus > 1 or available_gpus > 0 else 'CPU'}, with {tasks_per_gpu} parallel tasks per device")
    print(f"Input directory: {base_dir}")
    print(f"Output directory: {output_base_dir}")
    
    # Display GPU information
    if available_gpus > 0:
        for i in range(num_gpus):
            gpu_info = torch.cuda.get_device_properties(i)
            total_memory_gb = gpu_info.total_memory / (1024**3)
            print(f"GPU {i}: {gpu_info.name}, Memory: {total_memory_gb:.1f} GB")
    
    # Find all slice directories
    print("Searching all slice directories...")
    start_time = time.time()
    slice_dirs = get_slice_dirs(base_dir)
    print(f"Found {len(slice_dirs)} slice directories, search time: {time.time() - start_time:.2f} seconds")
    
    # Check which directories need processing
    print("Checking which directories need processing...")
    dirs_to_process = []
    
    for slice_dir in tqdm.tqdm(slice_dirs):
        parent_dir = os.path.dirname(slice_dir)
        case_id = os.path.basename(parent_dir)
        
        # Calculate new output path
        rel_path = os.path.relpath(parent_dir, base_dir)
        new_parent_dir = os.path.join(output_base_dir, rel_path)
        output_npy = os.path.join(new_parent_dir, f"{case_id}_biomedclip_features.npy")
        
        if not os.path.exists(output_npy):
            dirs_to_process.append(slice_dir)
    
    print(f"Total {len(slice_dirs)} directories, {len(dirs_to_process)} need processing")
    
    if len(dirs_to_process) == 0:
        print("All directories already processed, no need to process again")
        return
    
    # Assign directories to GPUs
    dirs_per_gpu = [[] for _ in range(num_gpus)]
    for i, dir_path in enumerate(dirs_to_process):
        gpu_id = i % num_gpus
        dirs_per_gpu[gpu_id].append(dir_path)
    
    for gpu_id in range(num_gpus):
        print(f"GPU {gpu_id} assigned {len(dirs_per_gpu[gpu_id])} directories")
    
    # Prepare multiprocessing parameters
    batch_args = [(gpu_id, dirs_per_gpu[gpu_id], base_dir, output_base_dir, tasks_per_gpu) for gpu_id in range(num_gpus)]
    
    # Use multiprocessing
    start_time = time.time()
    with Pool(processes=num_gpus) as pool:
        all_results = pool.map(process_gpu_batch, batch_args)
    
    # Merge results
    flat_results = [r for batch in all_results for r in batch]
    success_count = sum(1 for r in flat_results if r)
    failed_count = len(flat_results) - success_count
    
    process_time = time.time() - start_time
    print(f"\nAll processing complete: {success_count} directories successful, {failed_count} directories failed")
    print(f"Total processing time: {process_time:.2f} seconds, average per directory: {process_time/len(dirs_to_process):.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description='Extract BiomedCLIP features from CT slices')
    parser.add_argument('--base_dir', type=str, default="Your Input Directory",
                        help='Dataset base directory')
    parser.add_argument('--output_dir', type=str, default="Your Output Directory",
                        help='Feature output directory')
    parser.add_argument('--num_gpus', type=int, default=8,
                        help='Number of GPUs to use')
    parser.add_argument('--tasks_per_gpu', type=int, default=8,
                        help='Number of parallel tasks per GPU')
    parser.add_argument('--single_case', type=str, default="",
                        help='Process single case, leave empty to process all cases')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process a single case or the entire dataset
    if args.single_case:
        slice_dir = os.path.join(args.base_dir, args.single_case, "slices")
        
        # Create new output path
        rel_path = args.single_case
        new_parent_dir = os.path.join(args.output_dir, rel_path)
        os.makedirs(new_parent_dir, exist_ok=True)
        output_npy = os.path.join(new_parent_dir, f"{args.single_case}_biomedclip_features.npy")
        
        if not os.path.exists(slice_dir):
            print(f"Error: Directory {slice_dir} does not exist")
            return
        
        # Set device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Load model
        print("Loading BiomedCLIP model...")
        model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        model = model.to(device)
        model.eval()
        
        # Extract features
        extract_features(slice_dir, output_npy, model, preprocess, device)
        print("Feature extraction complete!")
    else:
        # Process the entire dataset using multiple GPUs
        process_dataset_with_multi_gpus(args.base_dir, args.output_dir, args.num_gpus, args.tasks_per_gpu)

if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' to solve CUDA initialization issues in multiprocessing
    if torch.cuda.is_available():
        multiprocessing.set_start_method('spawn', force=True)
    main()