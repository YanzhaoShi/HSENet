import os
import csv

import sys
sys.path.append("/home/ynwang/Yanzhaoshi/HSENet/Preprint")

from typing import Optional, List, Dict
import transformers
from LaMed.src.model.multimodal_encoder.vit import ViT
from LaMed.src.model.multimodal_encoder.vit import ViT_stage2
from LaMed.src.model.CLIP_stage1 import M3DCLIP_stage1, M3DCLIPConfig_stage1
from LaMed.src.model.CLIP_stage2 import M3DCLIP_stage2, M3DCLIPConfig_stage2
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig, LlamaForCausalLM
from dataclasses import dataclass, field
from LaMed.src.model.language_model import LamedLlamaForCausalLM, LamedPhi3ForCausalLM
from LaMed.src.train.lamed_trainer import LaMedTrainer
from LaMed.src.dataset.multi_dataset import ITRDataset

from LaMed.src.model.CLIP import M3DCLIP, M3DCLIPConfig
from transformers import BertTokenizer
import random
from safetensors.torch import load_file

import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from Bench.dataset.multi_dataset import CapDataset, CapDataset_CT_Rate
import evaluate

normal_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: f"{self.shape} {normal_repr(self)}"

is_use_single_GPU = True  # True False

if is_use_single_GPU:  # use single GPU for debugging
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '18025'
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    import torch.distributed as dist
    dist.init_process_group(backend='nccl')

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = True
torch._dynamo.config.disable = True
from typing import Tuple

import bitsandbytes as bnb
from peft import get_peft_model, LoraConfig, TaskType

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import evaluate

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    model_name_or_path: Optional[str] = field(default="/disk1/Data/Yanzhao/M3D_Model/Phi-4-mini-instruct", metadata={"help": "Path to the LLM or MLLM."})
    model_type: Optional[str] = field(default="phi3", metadata={"help": "llama2, phi3"})

    freeze_backbone: bool = field(default=False)
    pretrain_mllm: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=True, metadata={"help": "Used in pretrain: tune mm_projector and embed_tokens"})
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None, metadata={"help": "Path to pretrained mm_projector and embed_tokens."})
    
    # Choose a pretrained Vision ViT
    pretrained_visual_clip: str = field(default="/home/ynwang/Yanzhaoshi/HSENet/Model/stage2_clip")
    
    # Choose the trained LLM weights, including mm projector and LoRA
    resume_mllm_weights: str = field(default="/home/ynwang/Yanzhaoshi/HSENet/Model/trained_vlm_for_bimcv_r/pytorch_model.bin")

    # image
    image_channel: int = field(default=1)
    image_size: tuple = field(default=(32, 256, 256))
    patch_size: tuple = field(default=(4, 16, 16))

    # vision
    vision_tower: Optional[str] = field(default="vit_stage2_dual_encoders")
    remain_2d3d_ViT_type: Optional[str] = field(default='dual_vits', metadata={"help": "3d_vit, 2e3_vit, dual_vits"})
    vision_select_layer: Optional[int] = field(default=-1)
    vision_select_feature: Optional[str] = field(default="patch")
    freeze_vision_tower: bool = field(default=True)

    # projector
    mm_projector_type: Optional[str] = field(default='VisualPacker_3d_phi_v3', metadata={"help": "[baseline, VisualPacker_3d, VisualPacker_3d_phi_v3]."})
    use_parallel_projector: bool = field(default=True)  # True False
    proj_layer_type: str = field(default="mlp", metadata={"help": "Type of layer in projector. options: [linear, mlp]."})
    proj_layer_num: int = field(default=2, metadata={"help": "Number of layers in projector."})
    proj_pooling_type: str = field(default="spatial", metadata={"help": "Type of pooling in projector. options: [spatial, sequence]."})
    proj_pooling_size: int = field(default=2, metadata={"help": "Size of pooling in projector."})

    # segvol
    segmentation_module: str = field(default=None, metadata={"help": "segvol"})
    pretrain_seg_module: str = field(default=None, metadata={"help": "Pretrained segvol model."})

    max_length: int = field(default=800)
    max_new_tokens: int = field(default=512)  # 400 512

    top_p: float = field(default=None)
    temperature: float = field(default=1.0)
    do_sample: bool = field(default=False)

    proj_out_num: int = field(default=256)

    device: str = field(default="cuda", metadata={"help": ""})
    data_root: str = field(default="/disk1/Data/Yanzhao/M3D_Cap/M3D-Cap/", metadata={"help": ""})
    
    # Use BIMCV-R dataset
    cap_data_path: str = field(default="/disk1/Data/Yanzhao/BIMCV_R/curated_json_data/json_data/dataset.json", metadata={"help": "Path to caption data."})

    output_dir_eval: str = field(default="//home/ynwang/Yanzhaoshi/HSENet/Preprint/Bench/eval_caption_results/", metadata={"help": ""})
    file_name: str = field(default="eval_caption_ablation_dualvits_spatialpacker_BIMCV-R_v1.csv", metadata={"help": ""})
    

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # lora
    # lora_enable: bool = False
    lora_enable: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

    cache_dir: Optional[str] = field(default=None)
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(
        default=800,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    seed: int = 42
    ddp_backend: str = "nccl"
    ddp_timeout: int = 128000
    ddp_find_unused_parameters: bool = False

    # This is set up to facilitate debugging, pls config these in bash file in training.
    bf16: bool = True

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_mm_projector_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    # Process of elimination: LoRA only targets on LLM backbone
    ignore_keywords = ['vision_tower', 'mm_projector', 'embed_tokens', 'lm_head', 'seg_projector', 'seg_module']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in ignore_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    return list(lora_module_names)

def print_scores_for_all_captions(nlg_scores_for_report_generation, step_id):
    print(f"Step {step_id}: ")
    total_bleu1 = nlg_scores_for_report_generation[0]
    total_bleu2 = nlg_scores_for_report_generation[1]
    total_bleu3 = nlg_scores_for_report_generation[2]
    total_bleu4 = nlg_scores_for_report_generation[3]
    total_rougeL = nlg_scores_for_report_generation[4]
    total_meteor = nlg_scores_for_report_generation[5]
    total_bert_score = nlg_scores_for_report_generation[6]
    total_num = nlg_scores_for_report_generation[7]
    print(f"Mean bleu1: {total_bleu1/total_num:.4f} ({total_bleu1}/{total_num})")
    print(f"Mean bleu2: {total_bleu2/total_num:.4f} ({total_bleu2}/{total_num})")
    print(f"Mean bleu3: {total_bleu3/total_num:.4f} ({total_bleu3}/{total_num})")
    print(f"Mean bleu4: {total_bleu4/total_num:.4f} ({total_bleu4}/{total_num})")
    print(f"Mean rougeL: {total_rougeL/total_num:.4f} ({total_rougeL}/{total_num})")
    print(f"Mean meteor: {total_meteor/total_num:.4f} ({total_meteor}/{total_num})")
    print(f"Mean bert score: {total_bert_score/total_num:.4f} ({total_bert_score}/{total_num})")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels
        
import re

def truncate_string(input_str):
    input_str = "<|endoftext|>" + input_str.count("<im_patch>") * "<im_patch>"
    return input_str

def main():
    seed_everything(42)
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    vision_model_pretrained = AutoModel.from_pretrained(model_args.pretrained_visual_clip, trust_remote_code=True)
    rank0_print("="*20 + " Tokenizer preparation " + "="*20)
    # Load tokenizer from the given path with specified configurations
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Define and add special tokens
    special_token = {"additional_special_tokens": ["<im_patch>", "<bx_start>", "<bx_end>"]}
    tokenizer.add_special_tokens(
        special_token
    )
    tokenizer.add_tokens("[SEG]")

    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    if 'llama3' in model_args.model_type:
        tokenizer.eos_token_id = 128001
        tokenizer.pad_token = tokenizer.eos_token

    # Convert special tokens to token IDs and set related arguments
    model_args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    model_args.seg_token_id = tokenizer.convert_tokens_to_ids("[SEG]")
    model_args.vocab_size = len(tokenizer)
    rank0_print("seg_token_id: ", model_args.seg_token_id)
    rank0_print("vocab_size: ", model_args.vocab_size)

    rank0_print("="*20 + " Model preparation " + "="*20)
    if model_args.vision_tower is not None:
        if 'llama' in model_args.model_type:
            model = LamedLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                load_in_4bit=True,
                device_map="auto"
            )
            model = model.to('cuda')
        elif 'phi3' in model_args.model_type:
            model = LamedPhi3ForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                load_in_8bit=True,
                )
        else:
            raise ValueError(f"Unknown Model Type {model_args.model_type}")
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir
        )

    model.config.seg_token_id = model_args.seg_token_id
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    # initialize vision and seg modules on LLM
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args)
    if model_args.segmentation_module is not None:
        model.get_model().initialize_seg_modules(model_args=model_args)
    
    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
        try:
            for p in model.get_model().mm_projector2.parameters():
                p.requires_grad = True
        except:
            print("Without using mm_projector2.")

    model_args.num_new_tokens = 4
    model.initialize_vision_tokenizer(model_args, tokenizer)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        rank0_print("Adding LoRA adapters only on LLM.")
        model = get_peft_model(model, lora_config)

    rank0_print("="*20 + " Dataset preparation " + "="*20)

    print("proj_out_num: ", model_args.proj_out_num)

    visual_ckpt = vision_model_pretrained.state_dict()
    visual_ckpt = {k: v for k, v in visual_ckpt.items() if "language_encoder" not in k and "mm_vision_proj" not in k and "mm_language_proj" not in k and "logit_scale" not in k}  # 剩余288个key
    
    if model_args.vision_tower == "vit_single_3dvit_encoder" or model_args.vision_tower == "vit_stage2_dual_encoders":
        stage1_3d_vit_keys = [k for k in visual_ckpt.keys() if "stage1_pretrained_CLIP" in k]  # 剩余138个key
        print(f"pretrained stage1_3d_vit_keys: {len(stage1_3d_vit_keys)}")
        vision_encoder_keys_targets_stage1 = [k for k in model.state_dict().keys() if "vision_tower_stage1" in k]  # 数量138
        print(f"VLM stage1_3d_vit_keys: {len(vision_encoder_keys_targets_stage1)}")
        for i, k in enumerate(stage1_3d_vit_keys):
            try:
                model.state_dict()[vision_encoder_keys_targets_stage1[i]].copy_(visual_ckpt[k])
            except:
                print("not exist key: ", k)
    if model_args.vision_tower == "vit_single_2e3vit_encoder" or model_args.vision_tower == "vit_stage2_dual_encoders":
        stage2_2e3_vit_keys = [k for k in visual_ckpt.keys() if "stage1_pretrained_CLIP" not in k]  # 剩余150个key 比stage1多了12个key(10个slice_guided_attention*8和2个patch_score_proj)
        print(f"pretrained stage2_2e3_vit_keys: {len(stage2_2e3_vit_keys)}")
        vision_encoder_keys_targets_stage2 = [k for k in model.state_dict().keys() if "vision_tower_stage2" in k]  # 数量150
        print(f"VLM stage2_2e3_vit_keys: {len(vision_encoder_keys_targets_stage2)}")
        for i, k in enumerate(stage2_2e3_vit_keys):
            try:
                model.state_dict()[vision_encoder_keys_targets_stage2[i]].copy_(visual_ckpt[k])
            except:
                print("not exist key: ", k)

    print(f"load visual parameters model from pretrained pretrained CLIP ({model_args.pretrained_visual_clip}).")

    # Load the projector, LoRA, and ViT weights from the previously trained VLM.
    if model_args.resume_mllm_weights:
        projector_and_lora_checkpoints = torch.load(model_args.resume_mllm_weights, map_location='cpu')
        projector_checkpoint = {k: v for k, v in projector_and_lora_checkpoints.items() if 'mm_projector' in k}
        lora_checkpoint = {k: v for k, v in projector_and_lora_checkpoints.items() if 'lora' in k}
        save_params_checkpoint = {**projector_checkpoint, **lora_checkpoint}
        model.load_state_dict(save_params_checkpoint, strict=False)
        print(f"load projector, and lora parameters within pretrained VLM ({model_args.resume_mllm_weights}).")

    model.eval()
    for name, param in model.named_parameters():
        param.requires_grad = False
    model = model.to('cuda')

    from torch.utils.data import DataLoader
    import csv
    from Bench.dataset.multi_dataset import VQADataset
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
    device = "cuda"

    bleu = evaluate.load("bleu")
    bertscore = evaluate.load("bertscore")
    meteor = evaluate.load("meteor")
    rouge = evaluate.load("rouge")

    test_dataset = CapDataset_CT_Rate(model_args, tokenizer=tokenizer, mode='validation')
    batch_size_inference = 14
    test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size_inference,
            num_workers=24,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
    )  

    if not os.path.exists(model_args.output_dir_eval):
        os.makedirs(model_args.output_dir_eval)
    output_path = os.path.join(model_args.output_dir_eval, model_args.file_name)

    # Record the score of each category of questions
    nlg_scores_for_report_generation = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    total_num = test_dataloader.__len__()
    with open(output_path, mode='w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["Question", "Ground Truth", "pred", "bleu1", "bleu2", "bleu3", "bleu4", "rougeL", "meteor", "bert_f1"])
        sample_idx = 0
        for sample in tqdm(test_dataloader):
            question = sample["question"]
            question = [truncate_string(question[i]) + "Can you summarize with findings the images presented?" for i in range(len(question))]

            image = sample["image"].to(device=device)
            image_2d = sample["image_2d"].to(device=device)

            inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
            input_id = inputs['input_ids'].to(device=device)
            attention_mask = inputs['attention_mask'].to(device=device)

            with autocast():
                with torch.inference_mode():
                    generation = model.generate(image, input_id, image_2d=image_2d, attention_mask=attention_mask, max_new_tokens=model_args.max_new_tokens,
                                                        do_sample=model_args.do_sample, top_p=model_args.top_p,
                                                        temperature=model_args.temperature,
                                                        pad_token_id=model.config.pad_token_id, eos_token_id=model.config.eos_token_id)
            generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)

            answer = sample['answer']
            
            batch_size = image.size(0)

            for inner_batch_idx in range(batch_size):   
                sample_idx += 1

                result = dict()
                decoded_preds, decoded_labels = postprocess_text([generated_texts[inner_batch_idx]], [answer[inner_batch_idx]])
                bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels, max_order=4)
                result["bleu1"] = bleu_score["precisions"][0]  # BLEU-1
                result["bleu2"] = bleu_score["precisions"][1]  # BLEU-2
                result["bleu3"] = bleu_score["precisions"][2]  # BLEU-3
                result["bleu4"] = bleu_score["precisions"][3]  # BLEU-4

                rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels, rouge_types=['rougeL'])
                result["rougeL"] = rouge_score['rougeL']

                meteor_score = meteor.compute(predictions=decoded_preds, references=decoded_labels)
                result["meteor"] = meteor_score['meteor']

                bert_score = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
                result["bert_f1"] = sum(bert_score['f1']) / len(bert_score['f1'])

                nlg_scores_for_report_generation[0] += result["bleu1"]
                nlg_scores_for_report_generation[1] += result["bleu2"]
                nlg_scores_for_report_generation[2] += result["bleu3"]
                nlg_scores_for_report_generation[3] += result["bleu4"]
                nlg_scores_for_report_generation[4] += result["rougeL"]
                nlg_scores_for_report_generation[5] += result["meteor"]
                nlg_scores_for_report_generation[6] += result["bert_f1"]
                nlg_scores_for_report_generation[7] += 1

                if sample_idx % 100 == 0:
                    print_scores_for_all_captions(nlg_scores_for_report_generation, sample_idx)

                writer.writerow([question[inner_batch_idx], answer[inner_batch_idx], generated_texts[inner_batch_idx], result["bleu1"], result["bleu2"], result["bleu3"], result["bleu4"], result["rougeL"], result["meteor"], result["bert_f1"]])
        
        print_scores_for_all_captions(nlg_scores_for_report_generation, sample_idx)


if __name__ == "__main__":
    main()
       