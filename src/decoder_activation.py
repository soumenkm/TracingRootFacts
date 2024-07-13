import torch, torchinfo, json, sys, pickle, os, random
from typing import List, Tuple
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from decoder_models import LlamaModelForProbing, MixtralModelForProbing
from transformers import AutoTokenizer
from mask_dataset import MaskedDataset
from langcodes import Language
from itertools import combinations
import matplotlib.colors as mcolors
random.seed(42)

class Activation:
    
    def __init__(self, device: torch.device, mlama_dataset: MaskedDataset, tokenizer: AutoTokenizer, model: torch.nn.Module, name: str, is_load_model: bool = False):

        self.mlama_dataset = mlama_dataset
        self.tokenizer = tokenizer
        self.device = device
        self.model = model.to(self.device) if is_load_model else None
        self.output_type = "ff_out"
        self.name = name
        
        self.pkl_dir = Path(f"outputs/act/{self.name}")
        self.pkl_dir.mkdir(exist_ok=True)
        
        self.heatmap_dir = Path(f"outputs/activity_patterns/{self.name}")
        self.heatmap_dir.mkdir(exist_ok=True)
        
        self.prompt = ("Predict the correct text ONLY at masked position.\n" + 
            "Input: {input}\nPredicted masked output:")
        self.cache_data_dict = {}
        
    def get_dataset_for_lang(self, lang: str, num_of_examples: int, num_rels: int) -> dict:
        
        pkl_file_path = Path(self.pkl_dir, f"{lang}_activation_data.pkl")
        if Path.exists(pkl_file_path):
            data_dict = pickle.load(open(pkl_file_path, "rb"))
            print(f"The activation data is loaded from {pkl_file_path}")
            return data_dict
        
        if num_of_examples == -1:
            num_of_examples = len(self.mlama_dataset.uuid_info_all_lang)
        if num_rels == -1:
            num_rels = len(self.mlama_dataset.rels)
        
        rel_uri_list = random.sample(self.mlama_dataset.rels, k=num_rels)
            
        data_dict = {}
        with tqdm(self.mlama_dataset.uuid_info_all_lang.items(),
                  desc=f"Processing example: {lang}",
                  total=min(num_of_examples, len(self.mlama_dataset.uuid_info_all_lang)),
                  unit=" examples") as pbar:
            
            for count, (uuid, val) in enumerate(pbar):
                if lang in val.keys():
                    if val[lang]["rel_uri"] in rel_uri_list:
                        example = val[lang]["rel"].replace("[X]", str(val[lang]["sub"]))
                        new_mask = "[MASK]"
                        example = example.replace("[Y]", new_mask)
                        example = self.prompt.format(input=example)
                        inputs = self.tokenizer([example], return_tensors="pt")
                        
                        self.model.eval()
                        with torch.no_grad():
                            inputs = {k: v.to(self.device) for k, v in inputs.items()}
                            outputs = self.model(**inputs)
                        
                        ff_output = outputs[self.output_type] # (N, L, d)
                        avg_ff_output = ff_output.mean(dim=0) # (L, d)
                        
                        logits_output = outputs["logits"].argmax(dim=-1) # (N, d) -> (N,)
                        pred_text = self.tokenizer.decode(logits_output.tolist()).strip().lower()
                        true_text = val[lang]["obj"].strip().lower()
                        
                        new_data = {self.output_type: avg_ff_output,
                                    "inputs": example,
                                    "pred_text": pred_text,
                                    "true_text": true_text,
                                    "is_match": pred_text == true_text}
                        
                        data_dict[uuid] = {**val[lang], **new_data}
                        if count >= num_of_examples:
                            break
                else:
                    data_dict[uuid] = f"{lang} does not exist"
                    
        pickle.dump(data_dict, file=open(pkl_file_path, "wb"))
        return data_dict
    
    def load_pkl_activation_file(self, lang: str) -> dict:
        
        pkl_file_path = Path(self.pkl_dir, f"{lang}_activation_data.pkl")
                
        if pkl_file_path.exists():
            data_dict = pickle.load(open(pkl_file_path, "rb"))
            print(f"The activation data is loaded from {pkl_file_path}")
        else:
            raise FileNotFoundError(f"{pkl_file_path} does not exist!")
        
        self.cache_data_dict[lang] = data_dict
        
        return data_dict 
    
    def get_activity(self, data_dict: dict, fact_uuid: str) -> Tuple[torch.tensor, int]:
        
        act = data_dict[fact_uuid][self.output_type] # (L, d)
        rel = data_dict[fact_uuid]["rel_uri"]
        
        act_list = []
        for uuid, val in data_dict.items():
            if type(val) is not str:
                criteria = (val["rel_uri"] == rel) and (bool(val["is_match"]) == True)    
                if criteria:
                    act_list.append(val[self.output_type]) # List[R x (L, d)]
                    
        act_rel_tensor = torch.stack(act_list, dim=0) # (R, L, d)
        R = act_rel_tensor.shape[0]
        act_rel_tensor = act_rel_tensor.transpose(0, 1) # (L, R, d)
        act_rel_tensor = act_rel_tensor.mean(dim=1) # (L, d)
        activity = torch.abs(act_rel_tensor - act) # (L, d)
        
        return activity, R

    def get_binned_activity(self, act_data: torch.tensor) -> torch.tensor:
        
        num_bins = 16
        L, d = act_data.shape

        binned_act = torch.split(act_data, d//num_bins, dim=1) # Tuple[num_bins x (L, d')]
        binned_act = [i.mean(dim=-1) for i in binned_act] # List[num_bins x (L,)]
        binned_act = torch.stack(binned_act, dim=0) # (num_bins, L)
        binned_act = (binned_act - binned_act.min()) / (binned_act.max() - binned_act.min()) # (num_bins, L)
        binned_act = binned_act.transpose(0, 1) # (L, num_bins)
        binned_act = binned_act.flip(dims=[0]) # (L, num_bins) but 0 means top layer and L means bottom layer
        
        return binned_act
    
    def plot_activity(self, data_dict: dict, lang: str, fact_uuid: str) -> torch.tensor:

        plot_path = Path(self.heatmap_dir, lang, f"{fact_uuid}_activity_heatmap.png")
        
        act_data, R = self.get_activity(data_dict=data_dict, fact_uuid=fact_uuid) # (L, d)
        binned_act = self.get_binned_activity(act_data=act_data) # (L, num_bins)
        L = binned_act.shape[0]
        
        plt.figure(figsize=(8, 10))
        plt.imshow(binned_act.cpu(), aspect='auto', cmap='viridis_r')
        plt.colorbar(label='Activation Value')
        plt.xlabel('Bins')
        plt.ylabel('Transformer Layers (0: bottom)')
        plt.yticks(ticks=range(L), labels=reversed(range(L)))
        
        obj = str(self.mlama_dataset.uuid_info_all_lang[fact_uuid]["en"]["obj"])
        
        example = self.mlama_dataset.uuid_info_all_lang[fact_uuid]["en"]["rel"].replace("[X]", 
            str(self.mlama_dataset.uuid_info_all_lang[fact_uuid]["en"]["sub"]))
        title = (f'Neuron Activity Pattern for Fact Translated to Lang: ' + str(Language.get(lang).display_name()) + 
                f'\nFact: {example} ([Y]: {obj})' + f"\nAvg over {R} triplets of same relation with only correct predictions")
        plt.title(title)
        plt.savefig(str(plot_path), dpi=300)
        plt.close()
    
        return binned_act
    
    def plot_activity_for_1_lang(self, lang, num_uuids=10):
        
        save_dir = Path(self.heatmap_dir, lang)
        save_dir.mkdir(exist_ok=True)
        data_dict = self.load_pkl_activation_file(lang=lang)
        
        uuid_list = []
        for uuid, val in self.mlama_dataset.uuid_info_all_lang.items():
            if lang in val.keys():
                try:
                    if bool(data_dict[uuid]["is_match"]):
                        if "en" in self.mlama_dataset.uuid_info_all_lang[uuid].keys():
                            uuid_list.append(uuid)
                except KeyError:
                    pass

        if num_uuids == -1:
            num_uuids = len(uuid_list)
            
        for fact_uuid in random.sample(uuid_list, k=min(num_uuids, len(uuid_list))):
            self.plot_activity(data_dict=data_dict, lang=lang, fact_uuid=fact_uuid)
    
    def plot_activity_for_2_lang(self, lang1: str, lang2: str, is_right_only: bool = False):
        
        if is_right_only:
            save_dir = Path(self.heatmap_dir.parent, f"{self.name}_right_only")
        else:
            save_dir = self.heatmap_dir
            
        save_dir = Path(save_dir, f"{lang1}_{lang2}")
        save_dir.mkdir(exist_ok=True)
        
        if lang1 in self.cache_data_dict.keys():
            data_dict1 = self.cache_data_dict[lang1]
        else:
            data_dict1 = self.load_pkl_activation_file(lang=lang1)
        
        if lang2 in self.cache_data_dict.keys():
            data_dict2 = self.cache_data_dict[lang2]
        else:
            data_dict2 = self.load_pkl_activation_file(lang=lang2)
        
        uuid_list = []
        for uuid, val in self.mlama_dataset.uuid_info_all_lang.items():
            if (lang1 in val.keys()) and (lang2 in val.keys()):
                if (data_dict1[uuid]["is_match"] == True) and (data_dict2[uuid]["is_match"] == True):
                    if "en" in self.mlama_dataset.uuid_info_all_lang[uuid].keys():
                        uuid_list.append(uuid)
        
        # for fact_uuid in random.sample(uuid_list, k=min(50, len(uuid_list))):
        for fact_uuid in uuid_list[:10]:
            if is_right_only:
                type_pred = "only right" 
            else:
                type_pred = "both right and wrong"
                
            act_data1, R1 = self.get_activity(data_dict=data_dict1, fact_uuid=fact_uuid, is_right_only=is_right_only) # (L, d)
            binned_act1 = self.get_binned_activity(act_data=act_data1) # (L, num_bins)
            act_data2, R2 = self.get_activity(data_dict=data_dict2, fact_uuid=fact_uuid, is_right_only=is_right_only) # (L, d)
            binned_act2 = self.get_binned_activity(act_data=act_data2) # (L, num_bins)
            
            L = binned_act1.shape[0]
            fig, axes = plt.subplots(1, 3, figsize=(24, 10))

            # First language
            ax1 = axes[0]
            im1 = ax1.imshow(binned_act1.cpu(), aspect='auto', cmap='viridis_r')
            ax1.set_title(f'Language 1: {Language.get(lang1).display_name()}, R = {R1} triplets')
            ax1.set_xlabel('Bins')
            ax1.set_ylabel('Transformer Layers (0: bottom)')
            ax1.set_yticks(ticks=range(L))
            ax1.set_yticklabels(reversed(range(L)))
            fig.colorbar(im1, ax=ax1, label='Activation Value')

            # Second language
            ax2 = axes[1]
            im2 = ax2.imshow(binned_act2.cpu(), aspect='auto', cmap='viridis_r')
            ax2.set_title(f'Language 2: {Language.get(lang2).display_name()} , R = {R2} triplets')
            ax2.set_xlabel('Bins')
            ax2.set_ylabel('Transformer Layers (0: bottom)')
            ax2.set_yticks(ticks=range(L))
            ax2.set_yticklabels(reversed(range(L)))
            fig.colorbar(im2, ax=ax2, label='Activation Value')
            
            # Absolute difference
            ax3 = axes[2]
            abs_diff = torch.abs(binned_act1 - binned_act2)
            custom_cmap = mcolors.ListedColormap(['darkgreen', 'green', 'limegreen', 'palegreen', 'gold', 'orange', 'darkorange', 'red'])
            bounds = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
            norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)

            im3 = ax3.imshow(abs_diff.cpu(), aspect='auto', cmap=custom_cmap, norm=norm)
            ax3.set_title('Absolute Difference')
            ax3.set_xlabel('Bins')
            ax3.set_ylabel('Transformer Layers (0: bottom)')
            ax3.set_yticks(ticks=range(L))
            ax3.set_yticklabels(reversed(range(L)))
            fig.colorbar(im3, ax=ax3, label='Activation Value')

            example = self.mlama_dataset.uuid_info_all_lang[fact_uuid]["en"]["rel"].replace("[X]", 
                str(self.mlama_dataset.uuid_info_all_lang[fact_uuid]["en"]["sub"]))
            title = (f'Neuron Activity Pattern for Fact Translated to Lang: {Language.get(lang1).display_name()} and {Language.get(lang2).display_name()}' + 
                    f'\nFact in English: {example}' + f"\nAvg over R triplets of same relation having {type_pred} predictions")
            plt.suptitle(title)
            plt.tight_layout()
            plt.savefig(str(Path(save_dir, f"{fact_uuid}_activity_heatmap.png")), dpi=300)
            plt.close()
           
def main(model_name: str, device: torch.device):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if "llama" in model_name.lower():
        model = LlamaModelForProbing(tokenizer=tokenizer)
        name = "llama3-7B-instruct"
    elif "mixtral" in model_name.lower():
        model = MixtralModelForProbing(tokenizer=tokenizer)
        name = "mixtral-8x7B-Q4-instruct"
    else:
        raise NotImplementedError("Invalid model name!")
       
    mlama_dataset = MaskedDataset()
    activation = Activation(device=device, 
                            mlama_dataset=mlama_dataset, 
                            tokenizer=tokenizer, 
                            model=model, 
                            name=name,
                            is_load_model=False)
    language_codes = ['en', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'ru', 'zh', 'ja', 'ko']
    # for lang in language_codes:
    #     activation.get_dataset_for_lang(lang=lang, num_of_examples=-1, num_rels=10)
    # a = activation.load_pkl_activation_file(lang="en")
    for lang in language_codes:
        activation.plot_activity_for_1_lang(lang=lang, num_uuids=10)

    # a = activation.plot_activity(lang="ms", fact_uuid="b73bf6c6-3468-4ab8-9f4d-3c6e28259f07")
    # b = activation.plot_activity(lang="id", fact_uuid="b73bf6c6-3468-4ab8-9f4d-3c6e28259f07")
    
if __name__ == "__main__":
   
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    
    if torch.cuda.is_available():
        device_type = "cuda"
        print("Using GPU...")
        print(f"Total # of GPU: {torch.cuda.device_count()}")
        print(f"GPU Details: {torch.cuda.get_device_properties(device=torch.device(device_type))}")
    else:
        device_type = "cpu"
        print("Using CPU...")

    device = torch.device(device_type)
    models_list = ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mixtral-8x7B-Instruct-v0.1"]
    for model_name in models_list[:1]:
        main(model_name=model_name, device=device)