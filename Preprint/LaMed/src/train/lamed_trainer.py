import os
import torch
from transformers import Trainer
from transformers.utils import logging, SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from typing import Optional

logger = logging.get_logger(__name__)
TRAINING_ARGS_NAME = "training_args.bin"

class LaMedTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        if state_dict is None:
            state_dict = self.model.state_dict()

        logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
        projector_params = {name: param for name, param in self.model.named_parameters() if 'mm_projector' in name}
        lora_params = {name: param for name, param in self.model.named_parameters() if 'lora' in name}
        save_params = {**projector_params, **lora_params}
        torch.save(save_params, os.path.join(output_dir, WEIGHTS_NAME))
        print(f"Model weights saved to {output_dir}")

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))