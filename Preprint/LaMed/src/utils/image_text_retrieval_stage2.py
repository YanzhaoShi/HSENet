from typing import Optional
import transformers
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import norm
from transformers import BertTokenizer
import torch
from safetensors.torch import load_file
import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import json
import monai.transforms as mtf
import random
from typing import List
from LaMed.src.dataset.multi_dataset import CT_RateDataset
from LaMed.src.model.CLIP_stage1 import M3DCLIP_stage1, M3DCLIPConfig_stage1
from LaMed.src.dataset.multi_dataset import ITRDataset, CT_RateDataset_stage2
from LaMed.src.model.CLIP_stage2 import M3DCLIP_stage2, M3DCLIPConfig_stage2

normal_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: f"{self.shape} {normal_repr(self)}"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


device = torch.device("cuda") # or cpu

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

seed = 42

seed_everything(seed)

print("Seed is set to {}".format(seed))

model_path = "/path/to/stage2_CLIP_model"

model_name = "Stage_2"
print("Model path:", model_path)
tokenizer = AutoTokenizer.from_pretrained(
    "/path/to/bert-base-uncased",
    model_max_length=512,
    padding_side="right",
    use_fast=False
)
model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True
)
model = model.to(device=device)

def compute_accuracy(indices, ground_truth, k=5):
    correct_count = 0
    for i, pred_indices in enumerate(indices):
        if ground_truth[i] in pred_indices[:k]:
            correct_count += 1
    accuracy = correct_count / len(ground_truth)
    return accuracy


val_transform = mtf.Compose(
    [
        mtf.ToTensor(dtype=torch.float),
    ]
)
def truncate_text(input_text, max_tokens):
    def count_tokens(text):
        tokens = tokenizer.encode(text, add_special_tokens=True)
        return len(tokens)

    if count_tokens(input_text) <= max_tokens:
        return input_text

    sentences = input_text.split('.')

    selected_sentences = []
    current_tokens = 0

    if sentences:
        selected_sentences.append(sentences.pop(0))

    while current_tokens <= max_tokens and sentences:
        random_sentence = random.choice(sentences)
        new_tokens_len = count_tokens(random_sentence)
        if current_tokens + new_tokens_len <= max_tokens and random_sentence not in selected_sentences:
            selected_sentences.append(random_sentence)
            current_tokens += new_tokens_len
        else:
            sentences.remove(random_sentence)

    truncated_text = '.'.join(selected_sentences)
    return truncated_text

def multi_modal_retrieval(image_list, text_list, task, image_features, text_features, top_total_num=100, top_k=(5, 10, 50, 100)):
    results = []
    similarity = image_features @ text_features.T
    indices = torch.argsort(similarity, descending=True, dim=1)[:, :top_total_num]
    ground_truth = [i for i in range(image_features.size(0))]
    for topk in top_k:
        accuracy = compute_accuracy(indices, ground_truth, topk)
        print(f"Retrieval Top-{topk} Accuracy: {accuracy * 100:.4f}%")
        results.append(accuracy * 100)
        
    return results

def load_image(image_path):
    image = np.load(image_path)
    image_tensor = torch.from_numpy(image).unsqueeze(0)
    return image_tensor

def load_images_parallel(image_list):
    with ProcessPoolExecutor() as executor:
        images = list(tqdm(executor.map(load_image, image_list), total=len(image_list)))
    return images

def compute_all_sample(test_set):
    print("Test set:", test_set)
    with open("/path/to/data.json", "r") as f:
        data = json.load(f)[test_set]
    image_list, image_2d_list , text_list = [], [], []
    accession_number_list = []
    for item in data:
        image_list.append(item["image"])
        image_2d_list.append(item["biomedclip_features"])
        text_list.append(item["text"])
        accession_number = item["image"].split("/")[-1].split("_3D_features.npy")[0] + ".nii.gz"
        accession_number_list.append(accession_number)
    print("Number of samples:", len(image_list))
    image_input, image_2d_input, text_input = [], [], []
    for img_idx, image_path in tqdm(enumerate(image_list)):
        image = np.load(image_path)
        image_2d = np.load(image_2d_list[img_idx])
        image = torch.tensor(image).unsqueeze(0)
        image_2d = torch.tensor(image_2d).unsqueeze(0)
        image_input.append(image)
        image_2d_input.append(image_2d)
    for text in tqdm(text_list):
        raw_text = text
        raw_text = raw_text.replace('"', '')
        raw_text = raw_text.replace('\'', '')
        raw_text = raw_text.replace('(', '')
        raw_text = raw_text.replace(')', '')
        text = truncate_text(raw_text, 512)
        text_tensor = tokenizer(text, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
        input_id = text_tensor["input_ids"][0].unsqueeze(0)
        attention_mask = text_tensor["attention_mask"][0].unsqueeze(0)
        text_input.append((input_id, attention_mask))
    
    image_features, text_features = [], []
    for image_ipt_idx, image in tqdm(enumerate(image_input)):
        with torch.inference_mode():
            image_features.append(model.encode_image(image.to(device=device), image_2d_input[image_ipt_idx].to(device=device), None, None, image_list[image_ipt_idx])[:, 0].to(device="cpu"))
    for input_id, attention_mask in tqdm(text_input):
        with torch.inference_mode():
            model.to(device=device)
            text_features.append(model.encode_text(input_id.to(device=device), attention_mask.to(device=device))[0][:, 0].to(device="cpu"))
    
    image_features = torch.stack(image_features, dim=0).squeeze()
    text_features = torch.stack(text_features, dim=0).squeeze()
    print("Image-to-Text Retrieval:")
    acc_i2t = multi_modal_retrieval(image_list, text_list, "Image-to-Text Retrieval", image_features, text_features, top_total_num=100, top_k=(5, 10, 50, 100))

    print("Text-to-Image Retrieval:")
    acc_t2i = multi_modal_retrieval(text_list, image_list, "Text-to-Image Retrieval", text_features, image_features, top_total_num=100, top_k=(5, 10, 50, 100))

    print("Image-to-Image Retrieval:")
    
    filter_imgs = 0
    ratios_external = []
    image_data_for_second = []
    accs_for_second = []
    # Load the validation labels
    df = pd.read_csv("/path/to/valid_predicted_labels.csv")
    for k in tqdm(range(image_features.shape[0])):
        acc_second = accession_number_list[k]
        row_second = df[df['VolumeName'] == acc_second]
        num_path = np.sum(row_second.iloc[:, 1:].values[0])
        if num_path != 0:
            image_data_for_second.append(image_features[k])
            accs_for_second.append(accession_number_list[k])
        else:
            filter_imgs += 1
    print("Filter images:", filter_imgs)
    image_data_for_second = torch.stack(image_data_for_second, dim=0).squeeze()
    print(image_data_for_second.shape)
    for return_n in [1, 5, 10, 50]:
        for i in tqdm(range(image_features.shape[0])):
            first = image_features[i]
            acc_first = accession_number_list[i]
            row_first = df[df['VolumeName'] == acc_first]
            row_first = row_first.iloc[:, 1:].values[0]  # label
            first_expanded = first.unsqueeze(0)
            dot_products = torch.matmul(first_expanded.cuda(), image_data_for_second.cuda().T)
            magnitude_first = torch.norm(first).cuda()
            magnitudes_second = torch.norm(image_data_for_second, dim=1).cuda()
            cosine_similarities = dot_products.squeeze() / (magnitudes_second * magnitude_first)
            top_k_indices = torch.topk(cosine_similarities, return_n)[1].cpu()

            ratios_internal = []
            
            for index in top_k_indices:
                acc_second = accs_for_second[index]
                row_second = df[df['VolumeName'] == acc_second]
                row_second = row_second.iloc[:, 1:].values[0]
                ratio = calc_similarity(row_first, row_second)
                ratios_internal.append(ratio)

            ratios_external.append(np.mean(np.array(ratios_internal)))

        print("Image-to-Image Retrieval Top-{} Accuracy: {:.4f}%".format(return_n, np.mean(np.array(ratios_external)) * 100))


def calc_similarity(arr1, arr2):
    oneandone = 0
    oneorzero = 0
    zeroandzero = 0
    for k in range(len(arr1)):
        if arr1[k] == 0 and arr2[k] == 0:
            zeroandzero += 1
        if arr1[k] == 1 and arr2[k] == 1:
            oneandone += 1
        if arr1[k] == 0 and arr2[k] == 1:
            oneorzero += 1
        if arr1[k] == 1 and arr2[k] == 0:
            oneorzero += 1

    return (oneandone / (oneandone + oneorzero))

compute_all_sample("validation")
