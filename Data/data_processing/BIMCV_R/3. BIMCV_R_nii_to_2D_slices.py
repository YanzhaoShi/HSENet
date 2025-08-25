import os
import numpy as np
import torch
import nibabel as nib
from PIL import Image
import tqdm
import glob
import multiprocessing
from multiprocessing import Pool
import time
import argparse
import torch.nn.functional as F
import csv
import json
from datetime import datetime

'''
The function of this script is to process all nii.gz files to 2D slices using multiple GPUs.
You can specify the number of slices to extract from each 3D volume.
Steps:
1. Set the 'data_root' to the directory containing your NIfTI files.
2. Set the 'output_dir' to the directory where you want to save the processed 2D slices.
3. Set the 'num_slices' to the number of slices you want to extract from each 3D volume.
4. Set the 'gpu_ids' to the list of GPU IDs you want to use, e.g. [0, 1, 2, 3].
5. Run the script, it will automatically distribute the workload across the specified GPUs
'''


# Set multiprocessing start method to 'spawn' to solve CUDA initialization issues in multiprocessing
multiprocessing.set_start_method('spawn', force=True)

def extract_2d_slices_gpu(nii_file, output_dir, case_id, num_slices=32, gpu_id=0):
    """Using GPU to extract 2D slices, efficient version - optimized for large-scale processing"""
    try:
        print(f"GPU {gpu_id}: Processing file {os.path.basename(nii_file)}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Set GPU device
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(gpu_id)
        else:
            device = torch.device("cpu")
            print(f"GPU {gpu_id}: Warning: Falling back to CPU processing")
            
        # Read file size to select processing strategy
        file_size_mb = os.path.getsize(nii_file) / (1024 * 1024)
            
        # Load NIfTI file - with error handling
        try:
            # Use memory mapping to load nii file
            nii_img = nib.load(nii_file, mmap=True)
            
            if file_size_mb > 200:
                # Large file processing: load slices as needed
                shape = nii_img.shape
                z_indices = np.linspace(0, shape[2]-1, num_slices, dtype=int)
                
                # Pre-allocate GPU memory for all processed slices
                all_slices = torch.zeros((num_slices, shape[0], shape[1]), 
                                        device=device, dtype=torch.uint8)
                
                # Process slices one by one
                for i, z_idx in enumerate(z_indices):
                    slice_data = nii_img.dataobj[:, :, z_idx]
                    slice_tensor = torch.from_numpy(slice_data).float().to(device)
                    
                    # Slice-level normalization
                    slice_min = torch.min(slice_tensor)
                    slice_max = torch.max(slice_tensor)
                    
                    if slice_max > slice_min:
                        all_slices[i] = ((slice_tensor - slice_min) / (slice_max - slice_min) * 255).to(torch.uint8)
                    else:
                        all_slices[i] = torch.zeros_like(slice_tensor, dtype=torch.uint8, device=device)
                        
                    # Release temporary tensors
                    del slice_tensor
            else:
                # Small file processing: load all data at once
                nii_data = nii_img.get_fdata()
                volume = torch.from_numpy(nii_data).float().to(device)
                
                # Global normalization
                global_min = torch.min(volume)
                global_max = torch.max(volume)
                
                # Extract and process slices
                z_indices = torch.linspace(0, volume.shape[2]-1, num_slices, dtype=torch.int64, device=device)
                all_slices = torch.zeros((num_slices, volume.shape[0], volume.shape[1]), 
                                        device=device, dtype=torch.uint8)
                
                for i, z_idx in enumerate(z_indices):
                    slice_2d = volume[:, :, z_idx]
                    if global_max > global_min:
                        all_slices[i] = ((slice_2d - global_min) / (global_max - global_min) * 255).to(torch.uint8)
                    else:
                        all_slices[i] = torch.zeros_like(slice_2d, dtype=torch.uint8, device=device)
                
                # Release volume data
                del volume
        
        except RuntimeError as e:
            # Handle CUDA out of memory error
            if "CUDA out of memory" in str(e):
                print(f"GPU {gpu_id}: Out of memory, clearing cache and retrying...")
                torch.cuda.empty_cache()
                
                # Fall back to CPU processing
                print(f"GPU {gpu_id}: Falling back to CPU processing for file {os.path.basename(nii_file)}")
                
                # CPU processing logic
                nii_data = nii_img.get_fdata()
                shape = nii_data.shape
                z_indices = np.linspace(0, shape[2]-1, num_slices, dtype=int)
                
                for i, z_idx in enumerate(z_indices):
                    slice_2d = nii_data[:, :, int(z_idx)]
                    # Normalization on CPU
                    min_val = np.min(slice_2d)
                    max_val = np.max(slice_2d)
                    if max_val > min_val:
                        normalized = ((slice_2d - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                    else:
                        normalized = np.zeros_like(slice_2d, dtype=np.uint8)
                    
                    # Save slice - rotate 90 degrees counterclockwise
                    slice_img = Image.fromarray(normalized)
                    # Apply 90 degrees counterclockwise rotation
                    slice_img = slice_img.rotate(90, expand=True)
                    jpg_path = os.path.join(output_dir, f"slice_{i:03d}.jpg")
                    slice_img.save(jpg_path, quality=95)
                
                print(f"GPU {gpu_id}: Saved {num_slices} slices to: {output_dir} using CPU")
                return True
            else:
                raise e
                
        # Save all slices in batch
        all_slices_cpu = all_slices.cpu()
        del all_slices  # Release slice data on GPU
        
        for i in range(num_slices):
            slice_img = Image.fromarray(all_slices_cpu[i].numpy())
            # Apply 90 degrees counterclockwise rotation
            slice_img = slice_img.rotate(90, expand=True)
            jpg_path = os.path.join(output_dir, f"slice_{i:03d}.jpg")
            slice_img.save(jpg_path, quality=95)
        
        print(f"GPU {gpu_id}: Saved {num_slices} slices to: {output_dir}")
        return True
        
    except Exception as e:
        print(f"GPU {gpu_id} error when processing file {nii_file}: {str(e)}")
        import traceback
        traceback.print_exc()
        # Clean GPU memory when error occurs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False

def process_file(args):
    """Function to process a single file (for multi-GPU processing)"""
    nii_file, output_base_dir, num_slices, gpu_id, progress_file = args
    
    try:
        # Explicitly set CUDA device
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            # Only clean GPU memory when starting to process a new file
            torch.cuda.empty_cache()
        
        # Extract case_id from file path
        case_id = os.path.basename(nii_file).split('.')[0]
        
        # Keep CT structure for output directory, but put slices of each nii file in separate folder
        ct_dir = os.path.basename(os.path.dirname(os.path.dirname(nii_file)))
        # Modify output directory structure: use case_id as separate folder name
        output_dir = os.path.join(output_base_dir, ct_dir, case_id)
        
        # Optimize skip check - only check first and last file
        first_slice = os.path.join(output_dir, f"slice_000.jpg")
        last_slice = os.path.join(output_dir, f"slice_{num_slices-1:03d}.jpg")
        
        if os.path.exists(first_slice) and os.path.exists(last_slice):
            print(f"GPU {gpu_id}: Skipping already processed file: {nii_file}")
            return True
            
        result = extract_2d_slices_gpu(nii_file, output_dir, case_id, num_slices, gpu_id)
        
        # Update progress file
        if result:
            with open(progress_file, 'a') as f:
                f.write(f"{nii_file}\n")
                
        return result
    
    except Exception as e:
        print(f"GPU {gpu_id} unexpected error when processing file: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def process_bimcv_dataset(data_root, output_base_dir, num_slices=32, gpu_ids=None, batch_size=None):
    """
    Large-scale processing of BIMCV-R dataset
    
    Parameters:
    - data_root: Root directory containing BIMCV-R dataset
    - output_base_dir: Output directory
    - num_slices: Number of 2D slices to extract from each 3D volume
    - gpu_ids: List of GPU IDs to use, e.g. [0,1,2,3]
    - batch_size: Number of files to process at once, None means process all files
    """
    # Set GPUs to use
    if gpu_ids is None:
        gpu_ids = [0, 1, 2, 3]  # Default to use GPUs 0-3
    
    num_gpus = len(gpu_ids)
    print(f"Will use the following GPUs for processing: {gpu_ids} (total: {num_gpus})")
    
    # Record start time
    start_time = time.time()
    
    # Collect file paths
    print(f"Searching for .nii.gz files in directory {data_root}...")
    nii_files = glob.glob(os.path.join(data_root, "CT*", "ct", "*.nii.gz"))
    print(f"Found {len(nii_files)} nii.gz files")
    
    # Create progress tracking file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    progress_file = os.path.join(output_base_dir, f"progress_{timestamp}.txt")
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Check already processed files
    processed_files = set()
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            processed_files = set(line.strip() for line in f)
    
    # Filter out already processed files
    remaining_files = [f for f in nii_files if f not in processed_files]
    print(f"Need to process {len(remaining_files)} files, already processed {len(processed_files)} files")
    
    # Limit number of files (if needed)
    if batch_size is not None and batch_size < len(remaining_files):
        remaining_files = remaining_files[:batch_size]
        print(f"Batch processing mode: only processing {len(remaining_files)} files in this batch")
    
    # Prepare multi-GPU parameters - using specified GPU IDs
    args_list = [(nii_file, output_base_dir, num_slices, gpu_ids[i % num_gpus], progress_file) 
                for i, nii_file in enumerate(remaining_files)]
    
    # Use process pool for parallel processing
    print("Starting to process files...")
    process_start_time = time.time()
    
    # Use multiprocessing but limit number of processes to number of GPUs
    with Pool(processes=num_gpus) as pool:
        results = list(tqdm.tqdm(
            pool.imap(process_file, args_list),
            total=len(args_list),
            desc="Processing progress"
        ))
    
    # Count processing results
    success_count = sum(1 for r in results if r)
    failed_count = len(results) - success_count
    
    process_time = time.time() - process_start_time
    total_time = time.time() - start_time
    
    # Record results
    print(f"Processing complete: {success_count} files successful, {failed_count} files failed")
    if len(remaining_files) > 0:
        print(f"Processing time: {process_time:.2f} seconds, average per file: {process_time/len(remaining_files):.2f} seconds")
    print(f"Total run time: {total_time:.2f} seconds")
    
    # Save processing report
    report = {
        "timestamp": timestamp,
        "data_root": data_root,
        "output_dir": output_base_dir,
        "total_files": len(nii_files),
        "processed_files": len(processed_files) + success_count,
        "success_files": success_count,
        "failed_files": failed_count,
        "gpu_ids": gpu_ids,
        "num_gpus": num_gpus,
        "process_time": process_time,
        "total_time": total_time
    }
    
    report_file = os.path.join(output_base_dir, f"processing_report_{timestamp}.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"Processing report saved to: {report_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use GPU to convert 3D CT from BIMCV-R dataset to 2D slices")
    parser.add_argument("--data_root", type=str,  default = "./BIMCV_R/dataset", help="Root directory of BIMCV-R dataset")
    parser.add_argument("--output_dir", type=str, default = "./BIMCV_R/BIMCV_2D", help="Output directory")
    parser.add_argument("--num_slices", type=int, default=32, help="Number of 2D slices to extract from each 3D volume")
    parser.add_argument("--batch_size", type=int, default=None, help="Number of files to process at once, default processes all")
    parser.add_argument("--gpu_ids", type=int, nargs='+', default=[0, 1, 2, 3], help="GPU IDs to use, e.g. 0 1 2 3")
    
    args = parser.parse_args()
    
    print(f"Starting large-scale processing of BIMCV-R dataset...")
    print(f"Dataset directory: {args.data_root}")
    print(f"Output directory: {args.output_dir}")
    
    # Process dataset
    process_bimcv_dataset(
        args.data_root, 
        args.output_dir, 
        args.num_slices, 
        args.gpu_ids,
        args.batch_size
    )
    
    print("Dataset processing complete!")