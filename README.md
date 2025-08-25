# HSENet: Hybrid Spatial Encoding Network for 3D Medical Vision-Language Understanding

<font size=3><div align='center' > <a href=https://arxiv.org/abs/2506.09634>**Paper**</a> | [**Data**](#data) | [**Model**](#model) | [**Training**](#training) | [**inference**](#inference)</div></font>

HSENet introduces a novel approach to 3D medical vision-language understanding. It presents new paradigms for pre-training, a visual-language foundation model, and an efficient 3D token compression strategy.

## 🔥 Data Preparation

### Extract 3D Volume-Report Pairs from CT-RATE

> We use the **CT-RATE dataset** for medical image-text retrieval and report generation tasks. The data processing pipeline involves the following steps:

1. **Download the CT-RATE dataset** from [official source](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE).

2. **Normalize and standardize the 3D volume files** in `.nii` format, and convert them to `.npy` files:

   ```bash
   cd Data/data_processing/CT-RATE/
   python CT-RATE_nii_to_3D_volume_npy_file.py
   ```

3. **Extract 2D image slices from the 3D volumes** and store them:

   ```bash
   python CT-RATE_nii_to_2D_slices.py
   ```

4. **Normalize and standardize the extracted 2D slices**, then save them as `.npy` files:

   ```bash
   python CT-RATE_2D_to_npy_file.py
   ```

> For detailed instructions and file paths, please refer to the comments in each Python script.

### Extract 3D Volume-Report Pairs from BIMCV-R

1. **Download the BIMCV-R dataset** from [official source](https://huggingface.co/datasets/cyd0806/BIMCV-R).

2. **Process the data** following similar steps as for the CT-RATE dataset. The corresponding scripts are located in `Data/data_processing/BIMCV_R`.

### Extract VQA Samples from RadGenome

1. **Download the RadGenome dataset** from [official source](https://huggingface.co/datasets/RadGenome/RadGenome-ChestCT).

## 🔧 Requirements

To set up the environment, install the following dependencies:

```bash
torch==2.2.1
torchvision==0.17.1
transformers==4.49.0
open-clip-torch==2.24.0
```

For detailed environment setup, refer to the `requirements.txt` file.

We use Hugging Face’s **accelerate** for training. You can configure it with the following settings:

**File Path**:
`/home/user_name/.cache/huggingface/accelerate/default_config.yaml`

Example configuration:

```yaml
compute_environment: LOCAL_MACHINE
debug: true
distributed_type: MULTI_GPU
downcast_bf16: 'no'

dynamo_config:
    dynamo_backend: INDUCTOR
    dynamo_mode: default
    dynamo_use_dynamic: true
    dynamo_use_fullgraph: true

enable_cpu_affinity: false
machine_rank: 0
main_process_ip: localhost
main_process_port: 12376

main_training_function: main

num_machines: 1
num_processes: 8  # use 8 GPUs

rdzv_backend: static
same_network: true

tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false

use_cpu: false
```

---

## Model

| Model    | Download Link                                                                                                                                 |
|----------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| HSENet-CLIP | [HuggingFace](), [ModelScope]()    |
| HSENet-2E3-CLIP | [HuggingFace](), [ModelScope]()    |
| HSENet-Phi-4-4B (MRG) | [HuggingFace](), [ModelScope]()|
| HSENet-Phi-4-4B (VQA) | [HuggingFace](), [ModelScope]()|

**Notes:**

* **MRG** stands for **Medical Report Generation**.
* **HSENet-CLIP** and **HSENet-2E3-CLIP** are models trained in **Stage 1** and **Stage 2**, respectively.

## 🚀 Dual-stage Vision-Language Pre-training

The model is pretrained in two stages:

### Stage 1: Pre-training the 3D Vision Encoder

To train the Stage 1 model (**HSENet-CLIP**), run:

```bash
cd HSENet/Preprint/
nohup bash LaMed/script/train_clip_stage1.sh > train_stage1.log 2>&1 &
```

### Stage 2: Pre-training the 2D-Enhanced 3D Vision Encoder


To train the Stage 2 model (**HSENet-2E3-CLIP**), run:


```bash
cd HSENet/Preprint/
nohup bash LaMed/script/train_clip_stage2.sh > train_stage2.log 2>&1 &
```

---

## 🚀 MLLM Finetunning

### Medical Report Generation

To finetune the HSENet on report generation, run this command:

```bash
cd HSENet/Preprint/
nohup bash LaMed/script/train_vlm_mrg.sh > dualViTs_spatialpacker_mrg.log 2>&1 &
```

### Visual Question Answering (VQA)

To finetune the HSENet on VQA, run this command:

```bash
cd HSENet/Preprint/
nohup bash LaMed/script/train_vlm_vqa.sh > dualViTs_spatialpacker_vqa.log 2>&1 &
```

## 📊 Evaluation

### 3D Image-Text Retrieval

This evaluation includes:

* Report-to-Volume Retrieval
* Volume-to-Report Retrieval
* Volume-to-Volume Retrieval

To evaluate the **Stage 1** pre-trained model, run:

```bash
python Preprint/LaMed/src/utils/image_text_retrieval_stage1.py
```

To evaluate the **Stage 2** pre-trained model, run:

```bash
python Preprint/LaMed/src/utils/image_text_retrieval_stage2.py
```

### Medical Report Generation

To evaluate the performance of **HSENet** for report generation, run:

```bash
python Preprint/Bench/eval/eval_HSENet_CT_Rate_MRG.py
```

### Medical VQA

To evaluate the performance of **HSENet** for VQA, run:

```bash
python Preprint/Bench/eval/eval_HSENet_Rad_Geome_VQA.py
```

## Citation

If you find this project helpful, please cite our work:

```BibTeX
@article{shi2025hsenet,
  title={HSENet: Hybrid Spatial Encoding Network for 3D Medical Vision-Language Understanding},
  author={Shi, Yanzhao and Zhang, Xiaodan and Ji, Junzhong and Jiang, Haoning and Zheng, Chengxin and Wang, Yinong and Qu, Liangqiong},
  journal={arXiv preprint arXiv:2506.09634},
  year={2025}
}
```

## Acknowledgement

We would like to acknowledge open-source projects that have contributed to this work:

* [M3D](https://github.com/BAAI-DCAI/M3D)
* [Phi-4](https://huggingface.co/spaces/microsoft/phi-4-mini)