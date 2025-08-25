import os
import numpy as np
import torch
import nibabel as nib
import torch.nn.functional as F
import tqdm
import glob
import argparse
import multiprocessing
from multiprocessing import Pool
import time
import monai.transforms as mtf
from concurrent.futures import ThreadPoolExecutor
import gc

'''
The function of this script is to convert NIfTI files (.nii.gz) to 3D Volume NumPy arrays (.npy) using GPU acceleration.
It processes the files in batches, utilizing multiple GPUs and thread pools to optimize performance.
Steps:
1. Set the 'data_root' to the directory containing the NIfTI files.
2. Specify the 'output_dir' where the converted NumPy files will be saved.
3. Define the target shape for the output NumPy arrays (depth, height, width).
4. Optionally, limit the number of files to process with 'max_files'.
5. Optionally, specify the GPU IDs to use for processing.
6. Optionally, set the number of threads per GPU for parallel processing.
7. Optionally, define the maximum batch size in MB for memory management.
'''

multiprocessing.set_start_method('spawn', force=True)

def convert_nii_to_npy(nii_file, output_dir, target_shape=(32, 256, 256), gpu_id=0):
    
    try:
        # Create output directory
        case_id = os.path.basename(nii_file).split('.')[0]
        ct_dir = os.path.basename(os.path.dirname(os.path.dirname(nii_file)))
        sub_dir = os.path.join(output_dir, ct_dir)
        os.makedirs(sub_dir, exist_ok=True)
        output_path = os.path.join(sub_dir, f"{case_id}.npy")
        
   
        if os.path.exists(output_path):
            return True
            
        # Set GPU device
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(gpu_id)
       
        
        # Load NIfTI file
        nii_img = nib.load(nii_file, mmap=True)
        img_data = nii_img.get_fdata(dtype=np.float32)
        

        img_data_tensor = torch.tensor(img_data, dtype=torch.float32).to(device)
        
       
        hu_min, hu_max = -1000, 200
        img_data_tensor = torch.clamp(img_data_tensor, min=hu_min, max=hu_max)
        
    
        img_data = img_data_tensor.permute(2, 0, 1)
        
    
        tensor = img_data.unsqueeze(0)
        print(f"Tensor shape (batch dimension): {tensor.shape}")
        
        # Normalization processing
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        tensor = tensor - min_val
        normalized_tensor = tensor / torch.clamp(max_val - min_val, min=1e-8)
        
        
       
        transform = mtf.Compose([
            mtf.CropForeground(),  
            mtf.Resize(spatial_size=target_shape, mode="bilinear")
        ])
        
        # Apply transformation
        processed_tensor = transform(normalized_tensor)
        print(f"Shape after processing: {processed_tensor.shape}")
        # Convert to numpy and save
        np_tensor = processed_tensor.cpu().numpy()
        np.save(output_path, np_tensor)
        
        # Clean CPU and GPU memory - reduce cleaning calls
        del tensor, processed_tensor, img_data

        return True
            
    except RuntimeError as e:
        
        if "CUDA out of memory" in str(e):
            print(f"[GPU {gpu_id}] Out of memory, trying batch processing: {nii_file}")
            try:
                # Force GPU memory cleanup only when out of memory
                torch.cuda.empty_cache()
                gc.collect()
                
                # Load NIfTI file
                nii_img = nib.load(nii_file)
                img_data = nii_img.get_fdata(dtype=np.float32)
                
                # Get volume dimensions and calculate step size
                if img_data.shape[2] < img_data.shape[0]:
                    # Need to transpose
                    img_data = img_data.transpose(2, 0, 1)
                    
                D = img_data.shape[0]
                target_d = target_shape[0]
                
                # Calculate appropriate step size to get target number of slices
                step = max(1, D // target_d)
                
                # Initialize result list
                processed_slices = []
                
                # Reduce memory cleaning frequency
                slice_counter = 0
                
                # Batch processing
                for i in range(0, min(D, target_d * step), step):
                    if len(processed_slices) >= target_d:
                        break
                        
                    # Get current slice and move to GPU
                    slice_data = img_data[i]
                    slice_tensor = torch.tensor(slice_data, dtype=torch.float32).to(device)
                    
                    
                    slice_tensor = torch.clamp(slice_tensor, min=-1000, max=200)
                    
                    # Resize
                    slice_tensor = slice_tensor.unsqueeze(0).unsqueeze(0) 
                    resized_slice = F.interpolate(
                        slice_tensor, 
                        size=(target_shape[1], target_shape[2]),
                        mode='bilinear', 
                        align_corners=False
                    )
                    
                    # Remove extra dimensions
                    resized_slice = resized_slice.squeeze(0).squeeze(0)
                    
                    # Add to result list
                    processed_slices.append(resized_slice)  
                    slice_counter += 1
                    if slice_counter % 5 == 0:
                        del slice_tensor
                        torch.cuda.empty_cache()
                
                # Ensure correct number of slices
                while len(processed_slices) < target_d:
                    zero_slice = torch.zeros((target_shape[1], target_shape[2]), device=device)
                    processed_slices.append(zero_slice)
                
                # Stack slices to form final volume
                processed_slices = processed_slices[:target_d]  
                volume_tensor = torch.stack(processed_slices, dim=0)
                volume_tensor = volume_tensor.unsqueeze(0)  # Add batch dimension
                
            
                min_val = torch.min(volume_tensor)
                max_val = torch.max(volume_tensor)
                volume_tensor = volume_tensor - min_val
                normalized_tensor = volume_tensor / torch.clamp(max_val - min_val, min=1e-8)
                
                # Save as NPY
                np_tensor = normalized_tensor.cpu().numpy()
                np.save(output_path, np_tensor)
                
                # Clean up only once after batch processing is complete
                del volume_tensor, normalized_tensor, processed_slices
                
                return True
                
            except Exception as inner_e:
                print(f"[GPU {gpu_id}] Batch processing failed: {nii_file}, error: {inner_e}")
                torch.cuda.empty_cache()  # Still need cleaning when error occurs
                return False
        else:
            print(f"[GPU {gpu_id}] Processing error: {nii_file}, error: {e}")
            torch.cuda.empty_cache()  # Still need cleaning when error occurs
            return False
            
    except Exception as e:
        print(f"[GPU {gpu_id}] Processing error: {nii_file}, error: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Still need cleaning when error occurs
        return False

def process_file(args):
    """Wrapper function to process a single file"""
    nii_file, output_dir, target_shape, gpu_id, batch_size, batch_idx = args
    try:
        # Explicitly set GPU device and clean cache
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
        
        # Call conversion function
        return convert_nii_to_npy(nii_file, output_dir, target_shape, gpu_id)
    
    except Exception as e:
        print(f"[GPU {gpu_id}, Batch {batch_idx}] Error processing file: {str(e)}")
        return False

def process_gpu_batch(args):
    """Process a batch of files assigned to a single GPU"""
    gpu_id, files_batch, output_dir, target_shape, threads_per_gpu = args
    print(f"[GPU {gpu_id}] Starting to process {len(files_batch)} files, using {threads_per_gpu} threads")
    
    # Clean memory only once at the start of batch processing
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
        gc.collect()
    
    # Prepare task parameters
    tasks = []
    for i, nii_file in enumerate(files_batch):
        tasks.append((nii_file, output_dir, target_shape, gpu_id, len(files_batch), i))
    
    # Use optimized thread pool settings
    results = []
    with ThreadPoolExecutor(max_workers=threads_per_gpu) as executor:
        futures = [executor.submit(process_file, task) for task in tasks]
        
        # Use tqdm to track progress
        completed = 0
        for future in tqdm.tqdm(
            futures, 
            desc=f"[GPU {gpu_id}] Processing progress", 
            position=gpu_id,
            total=len(files_batch)
        ):
            results.append(future.result())
            
            # Clean memory only every 10 files
            completed += 1
            if completed % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Summarize results
    success_count = sum(1 for r in results if r)
    failed_count = len(results) - success_count
    print(f"[GPU {gpu_id}] Processing completed: {success_count} successful, {failed_count} failed")
    
    # Clean memory once at the end of batch processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    return results

def process_dataset(data_root, output_dir, target_shape=(32, 256, 256), gpu_ids=None, 
                   threads_per_gpu=1, max_files=None, batch_size_mb=4000):
    """Batch process all nii.gz files in directory using multiple GPUs and thread pools"""
    
    if gpu_ids is None:
        gpu_ids = [0, 1, 2, 3]  # Default to use first 4 GPUs
    
    num_gpus = len(gpu_ids)
    print(f"Will use the following GPUs: {gpu_ids} (total {num_gpus})")
    
    # Search for all nii.gz files
    print(f"Searching for nii.gz files in {data_root}...")
    nii_files = glob.glob(os.path.join(data_root, "CT*", "ct", "*.nii.gz"))
    print(f"Found {len(nii_files)} nii.gz files")
    
    # Limit file count (if needed)
    if max_files is not None:
        nii_files = nii_files[:max_files]
        print(f"Will only process the first {len(nii_files)} files")
    
    # Calculate file sizes and sort
    print("Getting file information based on file size...")
    nii_files_with_size = []
    for f in tqdm.tqdm(nii_files):
        # Check if target file already exists
        case_id = os.path.basename(f).split('.')[0]
        ct_dir = os.path.basename(os.path.dirname(os.path.dirname(f)))
        sub_dir = os.path.join(output_dir, ct_dir)
        output_path = os.path.join(sub_dir, f"{case_id}.npy")
        
        if not os.path.exists(output_path):
            size = os.path.getsize(f)
            nii_files_with_size.append((f, size))
    
    print(f"Files to process: {len(nii_files_with_size)}")
    
    if len(nii_files_with_size) == 0:
        print("No files to process, task completed!")
        return
    
    # Sort by size
    nii_files_with_size.sort(key=lambda x: x[1], reverse=True)
    
    # Print GPU memory info (for reference only)
    if torch.cuda.is_available():
        for gpu_id in gpu_ids:
            torch.cuda.set_device(gpu_id)
            free_memory, total_memory = torch.cuda.mem_get_info()
            free_memory_mb = free_memory / (1024 * 1024)
            print(f"GPU {gpu_id} available memory: {free_memory_mb:.2f} MB")
    
    # Simple average allocation of files to each GPU
    files_per_gpu = [[] for _ in range(num_gpus)]
    total_files = len(nii_files_with_size)
    
    # Calculate number of files to allocate to each GPU
    base_count = total_files // num_gpus
    remainder = total_files % num_gpus
    
    # Allocate files
    start_idx = 0
    for i in range(num_gpus):
        # Calculate how many files to allocate to current GPU
        count = base_count + (1 if i < remainder else 0)
        end_idx = start_idx + count
        
        # Assign files to current GPU
        for j in range(start_idx, end_idx):
            if j < total_files:
                files_per_gpu[i].append(nii_files_with_size[j][0])
        
        start_idx = end_idx
    
    # Print allocation
    for i in range(num_gpus):
        print(f"GPU {gpu_ids[i]} assigned {len(files_per_gpu[i])} files")
    
    # Prepare multiprocessing parameters
    batch_args = [(gpu_ids[i], files_per_gpu[i], output_dir, target_shape, threads_per_gpu) 
                  for i in range(num_gpus)]
    
    # Use multiprocessing for processing
    print("Starting to process files...")
    start_time = time.time()
    
    with Pool(processes=num_gpus) as pool:
        all_results = pool.map(process_gpu_batch, batch_args)
    
    # Merge all results
    flat_results = [r for batch in all_results for r in batch]
    success_count = sum(1 for r in flat_results if r)
    failed_count = len(flat_results) - success_count
    
    process_time = time.time() - start_time
    print(f"Processing completed: {success_count} files successful, {failed_count} failed")
    print(f"Total processing time: {process_time:.2f} seconds, average per file: {process_time/len(nii_files_with_size):.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NIfTI files (.nii.gz) to NumPy arrays (.npy) using GPU acceleration")
    parser.add_argument("--data_root", type=str, default="./BIMCV_R/dataset", 
                        help="Root directory containing nii.gz files")
    parser.add_argument("--output_dir", type=str, default="The path to save the output files", 
                        help="Directory for output npy files")
    parser.add_argument("--depth", type=int, default=32, help="Target depth")
    parser.add_argument("--height", type=int, default=256, help="Target height")
    parser.add_argument("--width", type=int, default=256, help="Target width")
    parser.add_argument("--max_files", type=int, default=None, help="Maximum number of files to process, defaults to all files")
    parser.add_argument("--gpu_ids", type=int, nargs='+', default=[1,2], help="GPU IDs to use")
    parser.add_argument("--threads_per_gpu", type=int, default=10, help="Number of threads per GPU")
    parser.add_argument("--batch_size_mb", type=int, default=4000, help="Maximum memory size (MB) per batch processing")
    
    args = parser.parse_args()
    
    target_shape = (args.depth, args.height, args.width)
    
    print(f"Starting to process NIfTI files...")
    print(f"Dataset directory: {args.data_root}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target shape: {target_shape}")
    print(f"Threads per GPU: {args.threads_per_gpu}")
    
    process_dataset(
        args.data_root,
        args.output_dir,
        target_shape=target_shape,
        gpu_ids=args.gpu_ids,
        threads_per_gpu=args.threads_per_gpu,
        max_files=args.max_files,
        batch_size_mb=args.batch_size_mb
    )
    
    print("Processing completed!")