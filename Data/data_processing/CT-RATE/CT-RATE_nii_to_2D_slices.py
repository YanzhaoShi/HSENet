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
import torch.nn.functional as F
import pandas as pd
import re

'''
The function of this script is to process all nii.gz files to 2D slices using multiple GPUs.
You can specify the number of slices to extract from each 3D volume.
Steps:
1. Set the 'nii_dir' to the directory containing your NIfTI files.
2. Set the 'output_base_dir' to the directory where you want to save the processed 2D slices.
3. Set the 'metadata_path' to the path of the metadata CSV file in CT-RaTE dataset.
4. Run the script, it will automatically distribute the workload across 8 GPUs.
'''

# Set multiprocessing start method to 'spawn' to solve CUDA initialization issues in multiprocessing
multiprocessing.set_start_method('spawn', force=True)

def read_nii_data(file_path):
    """Read NIfTI file data"""
    try:
        nii_img = nib.load(file_path)
        nii_data = nii_img.get_fdata()
        return nii_data
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def resize_array(array, current_spacing, target_spacing):
    """Resample array based on target pixel spacing"""
    # Calculate new dimensions
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    ]
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False)
    return resized_array

def normalize_image_gpu(img_tensor, device):
    """
    Normalize image to [0, 255] range (on GPU)
    
    Args:
        img_tensor: Input tensor, should already be on GPU
        device: GPU device to use
    
    Returns:
        Normalized GPU tensor (torch.uint8)
    """
    # Ensure input is FloatTensor
    if not img_tensor.is_floating_point():
        img_tensor = img_tensor.float()
        
    # Calculate global minimum and maximum values
    global_min = torch.min(img_tensor)
    global_max = torch.max(img_tensor)
    
    if global_max > global_min:
        normalized = ((img_tensor - global_min) / (global_max - global_min) * 255).to(torch.uint8)
    else:
        normalized = torch.zeros_like(img_tensor, dtype=torch.uint8, device=device)
        
    return normalized

def extract_float_from_spacing(spacing_str):
    """Extract float value from various spacing string formats"""
    # Use regex to extract the first floating point number
    match = re.search(r'([0-9]*\.?[0-9]+)', str(spacing_str))
    if match:
        return float(match.group(1))
    return 1.0  # Default value

def check_gpu_memory(gpu_id, threshold_percent=85):
    """
    Check memory usage of specified GPU, clear cache if above threshold
    
    Args:
        gpu_id: GPU ID
        threshold_percent: Memory usage threshold percentage, default 85%
    
    Returns:
        bool: Whether cleanup was performed
    """
    if not torch.cuda.is_available():
        return False
    
    # Get GPU total memory and allocated memory
    torch.cuda.set_device(gpu_id)
    total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
    allocated_memory = torch.cuda.memory_allocated(gpu_id)
    
    # Calculate usage percentage
    percent_used = (allocated_memory / total_memory) * 100
    
    # If above threshold, clear cache
    if percent_used >= threshold_percent:
        print(f"[GPU {gpu_id}] Memory usage at {percent_used:.1f}% (threshold: {threshold_percent}%), clearing cache")
        torch.cuda.empty_cache()
        new_allocated = torch.cuda.memory_allocated(gpu_id)
        print(f"[GPU {gpu_id}] Memory usage after clearing: {new_allocated / 1024**2:.2f} MB ({(new_allocated / total_memory) * 100:.1f}%)")
        return True
    return False



def extract_2d_slices_gpu(nii_file, output_dir, metadata_dict=None, num_slices=32, gpu_id=0):
    """
    Extract 2D slices from .nii.gz file using specified GPU acceleration and apply metadata processing
    All processes (including normalization) are performed on GPU
    
    Args:
        nii_file: NIfTI file path
        output_dir: Output directory
        metadata_dict: Dictionary containing metadata, indexed by filename
        num_slices: Number of slices to extract
        gpu_id: GPU ID to use
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if GPU is available, use specified GPU
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(gpu_id)
        else:
            device = torch.device("cpu")
            print("Warning: GPU not available, falling back to CPU processing")
        
        # Read NII file data
        img_data = read_nii_data(nii_file)
        if img_data is None:
            return False
            
        file_name = os.path.basename(nii_file)
        
        # Apply scaling factors from metadata (if provided)
        if metadata_dict is not None and file_name in metadata_dict:
            try:
                # Get metadata directly from dictionary, avoid lookup in DataFrame each time
                meta_row = metadata_dict[file_name]
                slope = float(meta_row["RescaleSlope"])
                intercept = float(meta_row["RescaleIntercept"])
                
                # Safely extract xy spacing and z spacing
                xy_spacing = extract_float_from_spacing(meta_row["XYSpacing"])
                z_spacing = extract_float_from_spacing(meta_row["ZSpacing"])
                
            except Exception as e:
                print(f"Error applying metadata: {e}, file: {file_name}")
                raise ValueError(f"Metadata processing failed: {file_name}")
        else:
            if metadata_dict is not None:
                print(f"Did not find {file_name} in metadata, using default values")
                raise ValueError(f"Missing metadata: {file_name}")
        
        # Define target spatial resolution
        target_x_spacing = 0.75
        target_y_spacing = 0.75
        target_z_spacing = 1.5
        
        # Ensure z-axis is the first dimension
        if img_data.shape[2] < img_data.shape[0]:
            img_data = img_data.transpose(2, 0, 1)
        
        # Check and clean GPU memory if needed
        check_gpu_memory(gpu_id)
        
        # Improvement: Convert data to GPU tensor immediately to speed up subsequent processing
        tensor = torch.tensor(img_data, device=device)
        
        tensor = slope * tensor + intercept
        tensor = torch.clamp(tensor, min=-1000, max=1000)
        
        tensor = tensor / 1000.0
        
        # Apply spatial resampling
        current = (z_spacing, xy_spacing, xy_spacing)
        target = (target_z_spacing, target_x_spacing, target_y_spacing)
        
        # Convert to PyTorch tensor and move to GPU
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        
        # Resample
        resized_tensor = resize_array(tensor, current, target)
        
        # Check again and clean GPU memory if needed
        check_gpu_memory(gpu_id)
        
        # Get z-axis length and select slices
        z_depth = resized_tensor.shape[2]  # Get depth directly from GPU tensor
        
        if z_depth >= num_slices:
            # Evenly select num_slices slice indices
            indices = torch.linspace(0, z_depth-1, num_slices, dtype=torch.long, device=device)
            
            # Pre-allocate GPU memory
            slices_tensor = torch.zeros((num_slices, resized_tensor.shape[3], resized_tensor.shape[4]), 
                                       dtype=resized_tensor.dtype, device=device)
            
            # Extract slices on GPU
            for i, idx in enumerate(indices):
                slices_tensor[i] = resized_tensor[0, 0, idx]
        else:
            # If there are fewer slices than num_slices, interpolate on GPU
            slices_tensor = F.interpolate(resized_tensor, 
                                         size=(num_slices, resized_tensor.shape[3], resized_tensor.shape[4]), 
                                         mode='trilinear', 
                                         align_corners=False)
            slices_tensor = slices_tensor[0, 0]  # Remove batch and channel dimensions
            
        # Release tensors no longer needed
        del tensor
        del resized_tensor
            
        # Third memory check
        check_gpu_memory(gpu_id)
            
        # Normalize and save slices - all on GPU
        for i in range(num_slices):
            # Normalize to [0, 255] range on GPU
            slice_normalized = normalize_image_gpu(slices_tensor[i], device)
            
            # Move back to CPU for saving as JPG
            slice_cpu = slice_normalized.cpu().numpy()
            
            # Save as JPG
            slice_img = Image.fromarray(slice_cpu)
            slice_img = slice_img.rotate(-90, expand=True)  # Negative angle means clockwise rotation
            jpg_path = os.path.join(output_dir, f"slice_{i:03d}.jpg")
            slice_img.save(jpg_path, quality=95)
        
        # Release all GPU tensors when finished
        del slices_tensor
        
        print(f"[GPU {gpu_id}] Saved {num_slices} slices to: {output_dir}")
        return True
    except Exception as e:
        print(f"[GPU {gpu_id}] Error processing file {nii_file}: {str(e)}")
        import traceback
        traceback.print_exc()
        # Clean GPU cache when error occurs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False
    
def process_file(args):
    """Function to process a single file (for multiprocessing)"""
    nii_file, output_base_dir, nii_dir, metadata_dict, num_slices, gpu_id = args
    
    rel_path = os.path.relpath(nii_file, nii_dir)
    case_id = os.path.basename(nii_file).split('.')[0]
    
    output_dir = os.path.join(output_base_dir, os.path.dirname(rel_path), case_id, "slices")
    
    # Skip if directory exists and has the correct number of files
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) == num_slices:
        print(f"[GPU {gpu_id}] Skipping already processed file: {nii_file}")
        return True
    
    # Use specified GPU for slice extraction
    result = extract_2d_slices_gpu(nii_file, output_dir, metadata_dict, num_slices, gpu_id)
    
    return result

def process_gpu_batch(batch_args):
    """Process a batch of files assigned to a specific GPU"""
    gpu_id, file_batch, output_base_dir, nii_dir, metadata_dict, num_slices = batch_args
    print(f"[GPU {gpu_id}] Starting to process {len(file_batch)} files")
    
    results = []
    
    # Set CUDA device
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        # Clean cache once at initialization
        torch.cuda.empty_cache()
        
        # Get GPU total memory to calculate percentages
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
        allocated_memory = torch.cuda.memory_allocated(gpu_id)
        print(f"[GPU {gpu_id}] Memory status: {allocated_memory / 1024**2:.2f} MB allocated ({(allocated_memory / total_memory) * 100:.1f}%)")
    
    for i, nii_file in enumerate(tqdm.tqdm(file_batch, desc=f"GPU {gpu_id} Processing files", 
                                         position=gpu_id)):
        args = (nii_file, output_base_dir, nii_dir, metadata_dict, num_slices, gpu_id)
        result = process_file(args)
        results.append(result)
        
        # Check memory status every 10 files
        if (i+1) % 10 == 0 and torch.cuda.is_available():
            # Check memory and clean if needed
            check_gpu_memory(gpu_id)
            
            # Display current memory status
            allocated_memory = torch.cuda.memory_allocated(gpu_id)
            percent_used = (allocated_memory / total_memory) * 100
            print(f"[GPU {gpu_id}] After processing {i+1}/{len(file_batch)} files:")
            print(f"[GPU {gpu_id}] Memory usage: {allocated_memory / 1024**2:.2f} MB ({percent_used:.1f}%)")
    
    success_count = sum(1 for r in results if r)
    failed_count = len(results) - success_count
    print(f"[GPU {gpu_id}] Processing complete: {success_count} files successful, {failed_count} files failed")
    
    return results

def distribute_files_across_gpus(nii_files, num_gpus):
    """
    Intelligently distribute files across different GPUs for more balanced workload
    
    Args:
        nii_files: List of file paths
        num_gpus: Number of GPUs
    
    Returns:
        list: List of files assigned to each GPU
    """
    files_per_gpu = [[] for _ in range(num_gpus)]
    
    # Use interleaved allocation so each GPU processes files of different sizes
    for i, file_path in enumerate(nii_files):
        gpu_id = i % num_gpus
        files_per_gpu[gpu_id].append(file_path)
    
    return files_per_gpu

def distribute_tasks_per_gpu(nii_files, num_gpus, tasks_per_gpu=2):
    """
    Assign multiple task batches for each GPU
    
    Args:
        nii_files: List of all file paths
        num_gpus: Number of GPUs
        tasks_per_gpu: Number of tasks per GPU
    
    Returns:
        list: List of (gpu_id, file_list) tuples
    """
    # Calculate total number of tasks
    total_tasks = num_gpus * tasks_per_gpu
    
    # Create task list
    tasks = []
    files_per_task = len(nii_files) // total_tasks
    
    for gpu_id in range(num_gpus):
        for task_idx in range(tasks_per_gpu):
            # Calculate global task index
            global_task_idx = gpu_id * tasks_per_gpu + task_idx
            
            # Calculate file range for this task
            start_idx = global_task_idx * files_per_task
            end_idx = start_idx + files_per_task if global_task_idx < total_tasks - 1 else len(nii_files)
            
            # Assign files
            task_files = nii_files[start_idx:end_idx]
            tasks.append((gpu_id, task_files))
    
    return tasks

def cached_find_nii_files(base_dir, cache_file=None):
    """Use caching mechanism to speed up file searching"""
    if cache_file is None:
        cache_file = os.path.join(os.path.dirname(base_dir), "nii_files_cache.txt")
    
    print(f"Searching for nii.gz files, using cache: {cache_file}")
    start_time = time.time()
    
    # Check if cache exists and is fresh
    cache_valid = False
    if os.path.exists(cache_file):
        cache_time = os.path.getmtime(cache_file)
        # Check directory modification time
        dir_time = os.path.getmtime(base_dir)
        
        # If cache is newer than directory, use the cache
        if cache_time > dir_time:
            try:
                with open(cache_file, 'r') as f:
                    files = [line.strip() for line in f if line.strip()]
                    if files:
                        cache_valid = True
                        search_time = time.time() - start_time
                        print(f"Loaded {len(files)} files from cache, took: {search_time:.2f} seconds")
                        return files
            except:
                print("Cache file read failed, searching again")
    
    # If no cache or invalid cache, perform search
    if not cache_valid:
        # Use find command for fast searching
        try:
            import subprocess
            cmd = f"find {base_dir} -name '*.nii.gz'"
            result = subprocess.check_output(cmd, shell=True, text=True)
            files = [line.strip() for line in result.split('\n') if line.strip()]
        except:
            # Fall back to glob
            files = glob.glob(os.path.join(base_dir, "**/*.nii.gz"), recursive=True)
        
        # Save to cache
        try:
            with open(cache_file, 'w') as f:
                for file in files:
                    f.write(f"{file}\n")
            print(f"File list cached to: {cache_file}")
        except Exception as e:
            print(f"Cache save failed: {e}")
    
    search_time = time.time() - start_time
    print(f"Found {len(files)} nii.gz files, search time: {search_time:.2f} seconds")
    return files


def process_dataset_with_8_gpus(nii_dir, output_base_dir, metadata_path=None, num_slices=32, tasks_per_gpu=2):
    """
    Process all nii.gz files in the dataset using 8 GPUs, intelligently managing GPU memory
    
    Args:
        nii_dir: Directory containing nii.gz files
        output_base_dir: Output directory
        metadata_path: Path to metadata CSV file
        num_slices: Number of 2D slices to extract from each 3D volume
    """
    # Get number of available GPUs
    available_gpus = torch.cuda.device_count()
    target_gpus = 8  # Target to use 8 GPUs
    
    if available_gpus == 0:
        print("No GPUs available, will use CPU processing")
        num_gpus = 1  # Use CPU
        print("Warning: Unable to allocate 8 GPUs, in CPU mode only 1 process will be used")
    elif available_gpus < target_gpus:
        print(f"Warning: Requested 8 GPUs, but only detected {available_gpus} GPUs")
        num_gpus = available_gpus
        print(f"Will use all {num_gpus} available GPUs")
    else:
        num_gpus = target_gpus
        print(f"Will use the specified {num_gpus} GPUs")
    
    # Display information for all GPUs to be used
    for i in range(num_gpus):
        if torch.cuda.is_available():
            gpu_info = torch.cuda.get_device_properties(i)
            total_memory_gb = gpu_info.total_memory / (1024**3)
            print(f"GPU {i}: {gpu_info.name}, Memory: {total_memory_gb:.1f} GB")
    
    # Read metadata (if provided) and create lookup dictionary
    metadata_dict = None
    if metadata_path and os.path.exists(metadata_path):
        try:
            # Read CSV once and load into memory
            metadata_df = pd.read_csv(metadata_path)
            print(f"Loaded metadata with {len(metadata_df)} records")
            
            # Create dictionary with filename as key for faster lookup
            metadata_dict = {}
            for _, row in metadata_df.iterrows():
                volume_name = row['VolumeName']
                metadata_dict[volume_name] = row
            
            print(f"Created metadata lookup dictionary with {len(metadata_dict)} records")
            # Delete DataFrame to free memory
            del metadata_df
            
        except Exception as e:
            print(f"Error reading metadata: {e}")
            print("Will proceed without metadata processing")
    
    # Collect all file paths
    start_time = time.time()
    print(f"Searching for .nii.gz files in directory {nii_dir}...")
    # nii_files = glob.glob(os.path.join(nii_dir, "**/*.nii.gz"), recursive=True)
    nii_files = cached_find_nii_files(nii_dir)  # Method 3
    search_time = time.time() - start_time
    print(f"Found {len(nii_files)} nii.gz files, search time: {search_time:.2f} seconds")
    
    print("Checking which files need processing...")
    files_to_process = []
    for nii_file in tqdm.tqdm(nii_files):
        rel_path = os.path.relpath(nii_file, nii_dir)
        case_id = os.path.basename(nii_file).split('.')[0]
        output_dir = os.path.join(output_base_dir, os.path.dirname(rel_path), case_id, "slices")
        
        # Need to process if directory doesn't exist or file count is incorrect
        if not os.path.exists(output_dir) or len(os.listdir(output_dir)) != num_slices:
            files_to_process.append(nii_file)
    print(f"Total {len(nii_files)} files, {len(files_to_process)} files need processing")
    
    if len(files_to_process) == 0:
        print("All files have been processed, no further processing needed")
        return
    # Use smarter method to allocate files to each GPU
    files_per_gpu = [[] for _ in range(num_gpus)]
    for i, file_path in enumerate(files_to_process):
        gpu_id = i % num_gpus
        files_per_gpu[gpu_id].append(file_path)
    
    for gpu_id in range(num_gpus):
        print(f"GPU {gpu_id} assigned {len(files_per_gpu[gpu_id])} files")
    # Prepare multiprocessing parameters
    batch_args = []
    for gpu_id in range(num_gpus):
        batch_args.append((gpu_id, files_per_gpu[gpu_id], output_base_dir, nii_dir, metadata_dict, num_slices))
    
    # Use multiprocessing (one process per GPU)
    start_time = time.time()
    with Pool(processes=num_gpus) as pool:
        all_results = pool.map(process_gpu_batch, batch_args)
    
    # Merge results
    flat_results = [r for batch in all_results for r in batch]
    success_count = sum(1 for r in flat_results if r)
    failed_count = len(flat_results) - success_count
    
    process_time = time.time() - start_time
    print(f"\nAll processing complete: {success_count} files successful, {failed_count} files failed")
    print(f"Total processing time: {process_time:.2f} seconds, average per file: {process_time/len(nii_files):.2f} seconds")
    
    # Report final memory status
    if torch.cuda.is_available():
        for gpu_id in range(num_gpus):
            torch.cuda.set_device(gpu_id)
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
            allocated_memory = torch.cuda.memory_allocated(gpu_id)
            percent_used = (allocated_memory / total_memory) * 100
            print(f"GPU {gpu_id} final memory: {allocated_memory / 1024**2:.2f} MB ({percent_used:.1f}%)")



def main():
    # Process all nii.gz files in the training directory
    nii_dir = "./CT-RATE/CT-RATE-new/dataset/train/"
    output_base_dir = "./CT-RATE/processed/processed_2D_slices_new"
    metadata_path = "./CT-RATE/CT-RATE-new/dataset/metadata/train_metadata.csv"
    # nii_dir = "./CT-RATE/CT-RATE-new/dataset/valid/"
    # output_base_dir = "./CT-RATE/processed/processed_2D_slices_new_validation"
    # metadata_path = "./CT-RATE/CT-RATE-new/dataset/metadata/validation_metadata.csv"
    print(f"Starting batch processing using 8 GPUs for directory: {nii_dir}")
    print(f"Output directory: {output_base_dir}")
    print(f"Metadata file: {metadata_path}")
    print(f"Note: Automatic memory monitoring enabled, cache will be cleared when memory usage exceeds 85%")
    
    # Use 8 GPUs
    process_dataset_with_8_gpus(nii_dir, output_base_dir, metadata_path, num_slices=32, tasks_per_gpu = 1)
    
    print("8-GPU batch processing complete!")

if __name__ == "__main__":
    main()