import torch, torchinfo, json, sys, pickle, os
from typing import List, Tuple
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from models import BertModelForProbing, XLMRobertaModelForProbing
from transformers import BertTokenizer, XLMRobertaTokenizer
from mask_dataset import MaskedDataset

class Activation:
    
    def __init__(self, device: torch.device, mlama_dataset: MaskedDataset, tokenizer: "Tokenizer", model: "Model", name: str, output_type: str = "ff_out"):
        """output_type in {"ff_out", "ff_act"}
        """
        self.mlama_dataset = mlama_dataset
        self.tokenizer = tokenizer
        self.device = device
        self.model = model.to(self.device)
        self.output_type = output_type
        self.name = name
        
    def get_dataset_for_lang(self, lang: str, num_of_examples: int, reload=True) -> dict:
        
        pkl_file_path = Path(f"outputs/act/{self.name}/{lang}_activation_data.pkl")
        
        if reload and Path.exists(pkl_file_path):
            data_dict = pickle.load(open(pkl_file_path, "rb"))
            print(f"The activation data is loaded from {pkl_file_path}")
            return data_dict
        
        data_dict = {}
        with tqdm(self.mlama_dataset.uuid_info_all_lang.items(),
                  desc=f"Processing example: {lang}",
                  total=min(num_of_examples, len(self.mlama_dataset.uuid_info_all_lang)),
                  unit=" examples") as pbar:
            
            for count, (uuid, val) in enumerate(pbar):
                if lang in val.keys():
                    example = val[lang]["rel"].replace("[X]", str(val[lang]["sub"]))
                    new_mask = (self.tokenizer.mask_token + " ") * len(self.tokenizer.tokenize(val[lang]["obj"]))
                    example = example.replace("[Y]", new_mask.strip())
                    inputs = self.tokenizer(example, truncation=True, return_tensors="pt")
                    
                    self.model.eval()
                    with torch.no_grad():
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        outputs = self.model(**inputs)
                    
                    ff_output_list = outputs[self.output_type] # List[12 x (b, T, d)]
                    mask_index_list = outputs["mask_index"] # List[b x List[z_b]]
                    assert all([i.shape[0] == 1 for i in ff_output_list])
                    assert len(mask_index_list) == 1
                    assert outputs["logits"].shape[0] == 1
                    avg_mask_ff_output_list = [i.squeeze(0)[mask_index_list[0], :].mean(dim=0, keepdim=False) for i in ff_output_list] # List[12 x (d,)]
                    
                    model_logit_output = outputs["logits"].argmax(dim=-1).squeeze(0) # (d,)
                    pred_token_ids = model_logit_output[mask_index_list[0]].tolist()
                    pred_token = self.tokenizer.convert_ids_to_tokens(pred_token_ids)
                    true_token = self.tokenizer.tokenize(val[lang]["obj"])      
                    
                    new_data = {self.output_type: avg_mask_ff_output_list,
                                "inputs": example,
                                "pred_token": pred_token,
                                "true_token": true_token,
                                "mask_index": mask_index_list[0],
                                "is_match": pred_token == true_token}
                    
                    data_dict[uuid] = {**val[lang], **new_data}
                    if count >= num_of_examples:
                        break
                else:
                    data_dict[uuid] = f"{lang} does not exist"
                    
        pickle.dump(data_dict, file=open(pkl_file_path, "wb"))
        return data_dict
        
def main(model_name: str, device: torch.device, num_of_examples: int=10):
       
    if "bert-" in model_name:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModelForProbing(tokenizer=tokenizer)
        name = "mbert"
    elif "xlm-roberta" in model_name:
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        model = XLMRobertaModelForProbing(tokenizer=tokenizer)
        name = "xlmr"
    else:
        raise NotImplementedError("Invalid model name!")
    
    mlama_dataset = MaskedDataset()
    actvation = Activation(device=device, mlama_dataset=mlama_dataset, tokenizer=tokenizer, model=model, name=name)
    
    for lang in mlama_dataset.langs:
        if not Path(f"outputs/act/{name}/{lang}_activation_data.pkl").exists():
            actvation.get_dataset_for_lang(lang=lang, num_of_examples=num_of_examples)
    
if __name__ == "__main__":
    cuda_ids = [3]
    cvd = ""
    for i in cuda_ids:
        cvd += str(i) + ","
        
    os.environ["CUDA_VISIBLE_DEVICES"] = cvd
    
    if torch.cuda.is_available():
        device_type = "cuda"
        print("Using GPU...")
        print(f"Total # of GPU: {torch.cuda.device_count()}")
        print(f"GPU Details: {torch.cuda.get_device_properties(device=torch.device(device_type))}")
    else:
        device_type = "cpu"
        print("Using CPU...")

    device = torch.device(device_type)
    models_list = ["bert-base-multilingual-cased", "xlm-roberta-large"]
    for model_name in models_list:
        main(model_name=model_name, device=device, num_of_examples=100000)