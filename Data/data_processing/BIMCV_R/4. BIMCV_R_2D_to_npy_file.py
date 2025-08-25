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
The function of this script is to process all 2D slices to npy files using multiple GPUs.
You can specify the number of parallel tasks per GPU.
Steps:
1. Set the 'input_dir' to the directory containing your 2D slice directories.
2. Set the 'output_dir' to the directory where you want to save the processed npy files.
3. Set the 'gpu_ids' to the list of GPU IDs you want to use.
4. Set the 'tasks_per_gpu' to the number of parallel tasks you want to run on each GPU.
5. Run the script, it will automatically distribute the workload across the specified
'''

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
        # Use trunk (backbone network) instead of complete encode_image to get richer features
        image_features = model.visual.trunk(batch_tensor)
    
    # Move features to CPU and convert to NumPy
    features_array = image_features.cpu().numpy()
    
    # Save features
    np.save(output_npy, features_array)
    print(f"Features saved to: {output_npy}")
    print(f"Feature shape: {features_array.shape}")
    
    return output_npy

def process_single_case(args):
    """Function to process a single CT case"""
    ct_dir, case_id, input_base_dir, output_base_dir, model, preprocess, device, task_id, gpu_id = args
    
    try:
        # Construct slice directory path
        slice_dir = os.path.join(input_base_dir, ct_dir, case_id)
        
        # Create output directory structure
        output_dir = os.path.join(output_base_dir, ct_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Output npy file path
        output_npy = os.path.join(output_dir, f"{case_id}_features.npy")
        
        # Skip if already processed
        if os.path.exists(output_npy):
            print(f"[GPU {gpu_id}, Task {task_id}] Skipping already processed: {ct_dir}/{case_id}")
            return True
        
        # Check if slice directory exists
        if not os.path.exists(slice_dir):
            print(f"[GPU {gpu_id}, Task {task_id}] Error: Directory {slice_dir} does not exist")
            return False
        
        # Check if there are slices
        slice_files = glob.glob(os.path.join(slice_dir, "slice_*.jpg"))
        if not slice_files:
            print(f"[GPU {gpu_id}, Task {task_id}] Error: No slice images found in {slice_dir}")
            return False
        
        # Extract features
        extract_features(slice_dir, output_npy, model, preprocess, device)
        return True
    
    except Exception as e:
        print(f"[GPU {gpu_id}, Task {task_id}] Error processing {ct_dir}/{case_id}: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_gpu_batch(batch_args):
    """Process a batch of directories assigned to a specific GPU"""
    gpu_id, cases_batch, input_base_dir, output_base_dir, tasks_per_gpu = batch_args
    print(f"[GPU {gpu_id}] Starting to process {len(cases_batch)} cases using {tasks_per_gpu} parallel tasks")
    
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
    num_threads = min(tasks_per_gpu, len(cases_batch))
    results = []
    
    # Use thread pool to process directories in parallel
    task_args = [(ct_dir, case_id, input_base_dir, output_base_dir, model, preprocess, device, i % num_threads, gpu_id) 
                for i, (ct_dir, case_id) in enumerate(cases_batch)]
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        futures = [executor.submit(process_single_case, arg) for arg in task_args]
        
        # Use tqdm to display progress
        for future in tqdm.tqdm(
            futures, 
            desc=f"GPU {gpu_id} Processing", 
            position=gpu_id,
            total=len(cases_batch)
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

def find_all_cases(input_base_dir):
    """Find all cases that need to be processed"""
    print(f"Searching for all cases in {input_base_dir}...")
    
    # Find all CT* directories
    ct_dirs = glob.glob(os.path.join(input_base_dir, "CT*"))
    
    all_cases = []
    
    # Traverse all CT directories, find case directories within
    for ct_dir in ct_dirs:
        ct_name = os.path.basename(ct_dir)
        case_dirs = [d for d in os.listdir(ct_dir) if os.path.isdir(os.path.join(ct_dir, d))]
        
        for case_id in case_dirs:
            all_cases.append((ct_name, case_id))
    
    print(f"Found {len(all_cases)} cases in {len(ct_dirs)} CT directories")
    return all_cases

def process_dataset_with_multi_gpus(input_base_dir, output_base_dir, gpu_ids=None, tasks_per_gpu=4):
    """
    Process the entire dataset using multiple GPUs, running multiple tasks on each GPU
    
    Args:
        input_base_dir: 2D slice input directory
        output_base_dir: Feature output directory
        gpu_ids: List of GPU IDs to use
        tasks_per_gpu: Number of parallel tasks per GPU
    """
    # Ensure output directory exists
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Set default GPUs
    if gpu_ids is None:
        gpu_ids = [0, 1, 2, 3]  # Default to use four GPUs 0-3
    
    # Check GPU availability
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    if available_gpus == 0:
        print("No GPUs available, will use CPU")
        gpu_ids = [-1]  # Use CPU
    else:
        # Filter out unavailable GPUs
        valid_gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id < available_gpus]
        if not valid_gpu_ids:
            print(f"Warning: None of the specified GPUs {gpu_ids} are available. Defaulting to GPU 0.")
            valid_gpu_ids = [0]
        elif len(valid_gpu_ids) < len(gpu_ids):
            print(f"Warning: Some specified GPUs are not available. Using {valid_gpu_ids}")
        gpu_ids = valid_gpu_ids
    
    num_gpus = len(gpu_ids)
    print(f"Will use {num_gpus} {'GPUs' if num_gpus > 1 and available_gpus > 0 else 'devices'}: {gpu_ids}, {tasks_per_gpu} parallel tasks per device")
    print(f"Input directory: {input_base_dir}")
    print(f"Output directory: {output_base_dir}")
    
    # Display GPU information
    if available_gpus > 0:
        for gpu_id in gpu_ids:
            gpu_info = torch.cuda.get_device_properties(gpu_id)
            total_memory_gb = gpu_info.total_memory / (1024**3)
            print(f"GPU {gpu_id}: {gpu_info.name}, Memory: {total_memory_gb:.1f} GB")
    
    # Find all cases
    start_time = time.time()
    all_cases = find_all_cases(input_base_dir)
    print(f"Found {len(all_cases)} cases in {time.time() - start_time:.2f} seconds")
    
    # Check which cases need processing
    print("Checking which cases need processing...")
    cases_to_process = []
    
    for ct_dir, case_id in tqdm.tqdm(all_cases):
        # Calculate output file path
        output_dir = os.path.join(output_base_dir, ct_dir)
        output_npy = os.path.join(output_dir, f"{case_id}_features.npy")
        
        # Check if already processed
        if not os.path.exists(output_npy):
            cases_to_process.append((ct_dir, case_id))
    
    print(f"Total {len(all_cases)} cases, {len(cases_to_process)} need processing")
    
    if not cases_to_process:
        print("All cases already processed")
        return
    
    # Assign cases to GPUs
    cases_per_gpu = {gpu_id: [] for gpu_id in gpu_ids}
    for i, case in enumerate(cases_to_process):
        idx = i % num_gpus
        gpu_id = gpu_ids[idx]
        cases_per_gpu[gpu_id].append(case)
    
    for gpu_id in gpu_ids:
        print(f"GPU {gpu_id} assigned {len(cases_per_gpu[gpu_id])} cases")
    
    # Prepare multiprocessing parameters
    batch_args = [(gpu_id, cases_per_gpu[gpu_id], input_base_dir, output_base_dir, tasks_per_gpu) 
                  for gpu_id in gpu_ids]
    
    # Use multiprocessing
    start_time = time.time()
    with Pool(processes=num_gpus) as pool:
        all_results = pool.map(process_gpu_batch, batch_args)
    
    # Merge results
    flat_results = [r for batch in all_results for r in batch]
    success_count = sum(1 for r in flat_results if r)
    failed_count = len(flat_results) - success_count
    
    process_time = time.time() - start_time
    print(f"\nAll processing complete: {success_count} cases successful, {failed_count} cases failed")
    print(f"Total processing time: {process_time:.2f} seconds, average per case: {process_time/len(cases_to_process):.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description='Extract BiomedCLIP features from CT slices')
    parser.add_argument('--input_dir', type=str, default="./BIMCV_R/BIMCV_2D",
                        help='Base directory containing 2D slices')
    parser.add_argument('--output_dir', type=str, default="The output directory",
                        help='Output directory for features')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0, 1, 2, 3],
                        help='GPU IDs to use (default: 0 1 2 3)')
    parser.add_argument('--tasks_per_gpu', type=int, default=1,
                        help='Number of parallel tasks per GPU')
    parser.add_argument('--single_case', nargs=2, metavar=('CT_DIR', 'CASE_ID'), 
                        help='Process single case (provide CT_DIR and CASE_ID)')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process a single case or the entire dataset
    if args.single_case:
        ct_dir, case_id = args.single_case
        
        # Construct slice directory path
        slice_dir = os.path.join(args.input_dir, ct_dir, case_id)
        
        # Create output directory structure
        output_dir = os.path.join(args.output_dir, ct_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Output npy file path
        output_npy = os.path.join(output_dir, f"{case_id}_features.npy")
        
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
        # Process the entire dataset using specified GPUs
        process_dataset_with_multi_gpus(args.input_dir, args.output_dir, args.gpu_ids, args.tasks_per_gpu)

if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' to solve CUDA initialization issues in multiprocessing
    if torch.cuda.is_available():
        multiprocessing.set_start_method('spawn', force=True)
    main()