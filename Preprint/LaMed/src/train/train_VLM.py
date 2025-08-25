import os
import logging

import sys
sys.path.append("/home/ynwang/Yanzhaoshi/HSENet/Preprint")

from typing import Optional, List, Dict
import numpy as np
from typing import Tuple
import torch
import transformers
from LaMed.src.model.multimodal_encoder.vit import ViT
from LaMed.src.model.multimodal_encoder.vit import ViT_stage2
from LaMed.src.model.CLIP_stage1 import M3DCLIP_stage1, M3DCLIPConfig_stage1
from LaMed.src.model.CLIP_stage2 import M3DCLIP_stage2, M3DCLIPConfig_stage2
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig, LlamaForCausalLM
from dataclasses import dataclass, field
from LaMed.src.dataset.multi_dataset import UniDatasets, CapDataset, TextDatasets, VQADataset, TextDatasets_CT_Rate, CapDataset_CT_Rate, VQADataset_CT_Rate
from LaMed.src.model.language_model import LamedLlamaForCausalLM, LamedPhi3ForCausalLM
from LaMed.src.train.lamed_trainer import LaMedTrainer
from LaMed.src.dataset.multi_dataset import ITRDataset

from LaMed.src.model.CLIP import M3DCLIP, M3DCLIPConfig
from transformers import BertTokenizer
import torch
from safetensors.torch import load_file

normal_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: f"{self.shape} {normal_repr(self)}"

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

is_use_single_GPU = True 

if is_use_single_GPU:
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1' 
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12161' 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import torch.distributed as dist
    dist.init_process_group(backend='nccl')

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = True
torch._dynamo.config.disable = True


import bitsandbytes as bnb
from peft import get_peft_model, LoraConfig, TaskType

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    model_name_or_path: Optional[str] = field(default="/disk1/Data/Yanzhao/M3D_Model/Phi-4-mini-instruct", metadata={"help": "Path to the LLM or MLLM."})
    
    ablation_type: Optional[str] = field(default="dualvits_spatialpacker")
    model_type: Optional[str] = field(default="phi3", metadata={"help": "llama2, phi3, llama3-8b"})

    freeze_backbone: bool = field(default=False)
    pretrain_mllm: Optional[str] = field(default=None)

    tune_mm_mlp_adapter: bool = field(default=True, metadata={"help": "Used in pretrain: tune mm_projector and embed_tokens"})
    is_pretraining: bool = field(default=False, metadata={"help": "data usage"})  # False True

    pretrain_mm_mlp_adapter: Optional[str] = field(default=None, metadata={"help": "Path to pretrained mm_projector and embed_tokens."})

    pretrained_visual_clip: str = field(default="/home/ynwang/Yanzhaoshi/HSENet/Model/stage2_clip")
    resume_mllm_weights: str = field(default=None)

    # image
    image_channel: int = field(default=1)
    image_size: tuple = field(default=(32, 256, 256))
    patch_size: tuple = field(default=(4, 16, 16))

    # vision
    vision_tower: Optional[str] = field(default="vit_stage2_dual_encoders") # 
    remain_2d3d_ViT_type: Optional[str] = field(default='dual_vits', metadata={"help": "3d_vit, 2e3_vit, dual_vits"})
    vision_select_layer: Optional[int] = field(default=-1)
    vision_select_feature: Optional[str] = field(default="patch")
    freeze_vision_tower: bool = field(default=True)

    # projector
    mm_projector_type: Optional[str] = field(default='VisualPacker_3d_phi_v3', metadata={"help": "[]."})
    use_parallel_projector: bool = field(default=True)  # True False
    proj_layer_type: str = field(default="mlp", metadata={"help": "Type of layer in projector. options: [linear, mlp]."})
    proj_layer_num: int = field(default=2, metadata={"help": "Number of layers in projector."})
    proj_pooling_type: str = field(default="spatial", metadata={"help": "Type of pooling in projector. options: [spatial, sequence]."})
    proj_pooling_size: int = field(default=2, metadata={"help": "Size of pooling in projector."})

    # segvol
    segmentation_module: str = field(default=None, metadata={"help": "segvol"})
    pretrain_seg_module: str = field(default=None, metadata={"help": "Pretrained segvol model."})



@dataclass
class DataArguments:
    data_root: str = field(default="/disk1/Data/Yanzhao/M3D_Cap/M3D-Cap/", metadata={"help": "Root directory for all data."})
    use_training_data: str = field(default="caption", metadata={"help": "caption, closedvqa, openvqa, closedvqa_and_caption"})

    # CT_Rate Dataset
    cap_data_path: str = field(default="/disk1/Data/Yanzhao/CT-RATE/ct_rate_dataset_ordered_with_biomedclip.json", metadata={"help": "Path to caption data."})
    
    # BIMCV_R Dataset
    # cap_data_path: str = field(default="/disk1/Data/Yanzhao/BIMCV_R/curated_json_data/json_data/dataset.json", metadata={"help": "Path to caption data."})
    
    # RadGeome VQA Dataset
    # cap_data_path: str = field(default="/disk1/Data/Yanzhao/CT-RATE/radgenome_vqa_location.json", metadata={"help": "Path to caption data."})

    # VQA data
    is_vqa_mark: bool = field(default=False)  # True False
    vqa_data_train_path: str = field(default="/", metadata={"help": "Path to training VQA data."})
    vqa_data_val_path: str = field(default="/", metadata={"help": "Path to validation VQA data."})
    vqa_data_test_path: str = field(default="/", metadata={"help": "Path to testing VQA data."})

    vqa_yn_data_train_path: str = field(default="/", metadata={"help": "Path to training VQA Yes or No data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # lora
    # lora_enable: bool = False
    lora_enable: bool = True
    lora_r: int = 16  # 4 16
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
    optim: str = field(default="adamw_torch")

    # This is set up to facilitate debugging, pls config these in bash file in training.
    bf16: bool = True
    output_dir: str = "/home/ynwang/Yanzhaoshi/HSENet/Preprint/LaMed/output/VLM_202508"
    save_step_names_list: List[int] = field(default_factory=lambda: [20000])
    num_train_epochs: float = 6
    per_device_train_batch_size: int = 1 
    per_device_eval_batch_size: int = 10
    gradient_accumulation_steps: int = 1
    evaluation_strategy: str = "steps"
    eval_accumulation_steps: int = 1
    eval_steps: float = 0.04
    save_strategy: str = "steps"
    save_steps: int = 2000 
    save_total_limit: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: float = 10
    gradient_checkpointing: bool = True # train fast
    dataloader_pin_memory: bool = True # fast
    dataloader_num_workers: int = 0
    report_to: str = "tensorboard"


def compute_metrics(eval_preds):
    labels_ids = eval_preds.label_ids
    pred_ids = eval_preds.predictions

    labels = labels_ids[:, 1:]
    preds = pred_ids[:, :-1]

    labels_flatten = labels.reshape(-1)
    preds_flatten = preds.reshape(-1)
    valid_indices = np.where(labels_flatten != -100)
    filtered_preds = preds_flatten[valid_indices]
    filtered_labels = labels_flatten[valid_indices]
    acc_score = sum(filtered_preds==filtered_labels) / len(filtered_labels)

    return {"accuracy": acc_score}

def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


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



def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save projector and embed_tokens in pretrain
        keys_to_match = ['mm_projector', 'embed_tokens']

        weight_to_save = get_mm_projector_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

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

@dataclass
class DataCollator:
    def __init__(self, seg_enable):
        self.seg_enable = seg_enable
    def __call__(self, batch: list) -> dict:
        if self.seg_enable:
            images, input_ids, labels, attention_mask, segs = tuple(
                [b[key] for b in batch] for key in ('image', 'input_id', 'label', 'attention_mask', 'seg'))

            images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
            input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
            labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
            attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)

            for i, seg in enumerate(segs):
                if seg.sum() == 0:
                    segs[i] = torch.zeros((1, 1, 32, 256, 256))
                else:
                    segs[i] = seg.unsqueeze(0)
            segs = torch.cat(segs, dim=0)

            return_dict = dict(
                images=images,
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                segs=segs,
            )
        else:
            images, input_ids, labels, attention_mask, images_2d = tuple(
                [b[key] for b in batch] for key in ('image', 'input_id', 'label', 'attention_mask', 'image_2d'))

            images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
            input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
            labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
            attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)
            images_2d = torch.cat([_.unsqueeze(0) for _ in images_2d], dim=0)

            return_dict = dict(
                images=images,
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                images_2d=images_2d,
            )

        return return_dict

def main():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # ablation_type
    print("ablation_type: ", model_args.ablation_type)
    # vision_tower
    print("vision_tower: ", model_args.vision_tower)
    # mm_projector_type proj_layer_type
    print("mm_projector_type: ", model_args.mm_projector_type)
    print("proj_layer_type: ", model_args.proj_layer_type)
    # model_max_length
    print("model_max_length: ", training_args.model_max_length)
    # cap_data_path
    print("cap_data_path: ", data_args.cap_data_path)

    local_rank = training_args.local_rank
    vision_model_pretrained = AutoModel.from_pretrained(model_args.pretrained_visual_clip)
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

    model.enable_input_require_grads()
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

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

    if model_args.pretrain_mllm:
        ckpt = torch.load(model_args.pretrain_mllm, map_location="cpu")
        model.load_state_dict(ckpt, strict=True)
        rank0_print("load pretrained MLLM weights.")

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

        for n, p in model.named_parameters():
            if any(
                    [x in n for x in ['mm_projector']]
            ):
                p.requires_grad = True


    rank0_print("="*20 + " Dataset preparation " + "="*20)
    data_args.max_length = training_args.model_max_length
    if model_args.vision_tower == "vit_stage2_dual_encoders":
        if model_args.use_parallel_projector:
            data_args.proj_out_num = model.get_model().mm_projector.proj_out_num + model.get_model().mm_projector2.proj_out_num
        else:
            data_args.proj_out_num = model.get_model().mm_projector.proj_out_num * 2
    else:
        data_args.proj_out_num = model.get_model().mm_projector.proj_out_num
    rank0_print("vision tokens output from projector: ", data_args.proj_out_num)
    data_args.seg_enable = hasattr(model.get_model(), "seg_module")

    if model_args.tune_mm_mlp_adapter and not model_args.is_pretraining:
        train_dataset = TextDatasets_CT_Rate(data_args, tokenizer, mode='train')
    else:
        train_dataset = UniDatasets(data_args, tokenizer, mode='train')

    if data_args.use_training_data == "caption":
        eval_dataset = CapDataset_CT_Rate(data_args, tokenizer, mode='validation')
    elif data_args.use_training_data == "openvqa":
        eval_dataset = VQADataset_CT_Rate(data_args, tokenizer, mode='validation')

    data_collator = DataCollator(data_args.seg_enable)

    rank0_print("="*20 + " Training " + "="*20)

    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True

    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")

    visual_ckpt = vision_model_pretrained.state_dict()
    visual_ckpt = {k: v for k, v in visual_ckpt.items() if "language_encoder" not in k and "mm_vision_proj" not in k and "mm_language_proj" not in k and "logit_scale" not in k}
    if model_args.vision_tower == "vit_single_3dvit_encoder" or model_args.vision_tower == "vit_stage2_dual_encoders":
        stage1_3d_vit_keys = [k for k in visual_ckpt.keys() if "stage1_pretrained_CLIP" in k] 
        if stage1_3d_vit_keys == []:
            print("No stage1_pretrained_CLIP in visual_ckpt.")
            print("Use vision_encoder layers as stage1_3d_vit_keys.")
            stage1_3d_vit_keys = [k for k in visual_ckpt.keys() if "vision_encoder" in k] 
        print(f"pretrained stage1_3d_vit_keys: {len(stage1_3d_vit_keys)}")
        vision_encoder_keys_targets_stage1 = [k for k in model.state_dict().keys() if "vision_tower_stage1" in k]
        print(f"VLM stage1_3d_vit_keys: {len(vision_encoder_keys_targets_stage1)}")
        for i, k in enumerate(stage1_3d_vit_keys):
            try:
                model.state_dict()[vision_encoder_keys_targets_stage1[i]].copy_(visual_ckpt[k])
            except:
                print("not exist key: ", k)
    if model_args.vision_tower == "vit_single_2e3vit_encoder" or model_args.vision_tower == "vit_stage2_dual_encoders":
        stage2_2e3_vit_keys = [k for k in visual_ckpt.keys() if "stage1_pretrained_CLIP" not in k] 
        print(f"pretrained stage2_2e3_vit_keys: {len(stage2_2e3_vit_keys)}")
        vision_encoder_keys_targets_stage2 = [k for k in model.state_dict().keys() if "vision_tower_stage2" in k]
        print(f"VLM stage2_2e3_vit_keys: {len(vision_encoder_keys_targets_stage2)}")
        for i, k in enumerate(stage2_2e3_vit_keys):
            try:
                model.state_dict()[vision_encoder_keys_targets_stage2[i]].copy_(visual_ckpt[k])
            except:
                print("not exist key: ", k)
    print(f"load visual parameters model from pretrained pretrained CLIP ({model_args.pretrained_visual_clip}).")

    if model_args.resume_mllm_weights:
        projector_and_lora_checkpoints = torch.load(model_args.resume_mllm_weights, map_location='cpu')
        projector_checkpoint = {k: v for k, v in projector_and_lora_checkpoints.items() if 'mm_projector' in k}
        lora_checkpoint = {k: v for k, v in projector_and_lora_checkpoints.items() if 'lora' in k}
        save_params_checkpoint = {**projector_checkpoint, **lora_checkpoint}
        model.load_state_dict(save_params_checkpoint, strict=False)
        print(f"load projector, and lora parameters within pretrained VLM ({model_args.resume_mllm_weights}).")

    model.print_trainable_parameters()

    from transformers import TrainerCallback
    TRAINING_ARGS_NAME = "training_args.bin"
    class CustomSaveCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            """
            在每个 step 结束时调用。
            """
            if state.global_step in args.save_step_names_list:
                save_idx = state.global_step
                output_dir = args.output_dir 
                save_model_path = output_dir + f"/saved_{save_idx}"
                model = kwargs['model'] 

                os.makedirs(save_model_path, exist_ok=True)
                projector_params = {name: param for name, param in model.named_parameters() if 'mm_projector' in name}
                lora_params = {name: param for name, param in model.named_parameters() if 'lora' in name}
                save_params = {**projector_params, **lora_params}
                torch.save(save_params, os.path.join(save_model_path, 'pytorch_model.bin'))
                print(f"Model weights saved to {save_model_path}")

                torch.save(args, os.path.join(save_model_path, "training_args.bin"))

                print(f"step {save_idx} saved to {save_model_path}")   

    trainer = LaMedTrainer(
                            model=model.to('cuda'),
                            args=training_args,
                            data_collator=data_collator,
                            train_dataset=train_dataset,
                            eval_dataset=eval_dataset,
                            compute_metrics=compute_metrics,
                            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                      )

    trainer.train()
    trainer.save_state()
    model.config.use_cache = True

    rank0_print("="*20 + " Save model " + "="*20)
    if training_args.lora_enable:
        projector_params = {name: param for name, param in model.named_parameters() if 'mm_projector' in name}
        lora_params = {name: param for name, param in model.named_parameters() if 'lora' in name}
        save_params = {**projector_params, **lora_params}
        torch.save(save_params, os.path.join(training_args.output_dir, 'mm_projector_and_lora.bin'))
        print(f"Model weights saved to {training_args.output_dir}")
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    main()
