import torch, torchinfo, json, sys, pickle, os, random
from typing import List, Tuple
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from encoder_models import BertModelForProbing, XLMRobertaModelForProbing
from transformers import BertTokenizer, XLMRobertaTokenizer
from mask_dataset import MaskedDataset
from langcodes import Language
from itertools import combinations
import matplotlib.colors as mcolors
random.seed(42)

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
        self.pkl_dir = Path(f"outputs/act/{self.name}")
        self.heatmap_dir = Path(f"outputs/activity_patterns/{self.name}")
        self.cache_data_dict = {}
        
    def get_dataset_for_lang(self, lang: str, num_of_examples: int) -> dict:
        
        pkl_file_path = Path(self.pkl_dir, f"{lang}_activation_data.pkl")
        if Path.exists(pkl_file_path):
            data_dict = pickle.load(open(pkl_file_path, "rb"))
            print(f"The activation data is loaded from {pkl_file_path}")
            return data_dict
        
        if num_of_examples == -1:
            num_of_examples = len(self.mlama_dataset.uuid_info_all_lang)
            
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
                    
                    ff_output_list = outputs[self.output_type] # List[L x (b, T, d)]
                    mask_index_list = outputs["mask_index"] # List[b x List[z_b]]
                    assert all([i.shape[0] == 1 for i in ff_output_list])
                    assert len(mask_index_list) == 1
                    assert outputs["logits"].shape[0] == 1
                    avg_mask_ff_output_list = [i.squeeze(0)[mask_index_list[0], :].mean(dim=0, keepdim=False) for i in ff_output_list] # List[L x (d,)]
                    
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
    
    def load_pkl_activation_file(self, lang: str) -> dict:
        
        pkl_file_path = Path(self.pkl_dir, f"{lang}_activation_data.pkl")
                
        if pkl_file_path.exists():
            data_dict = pickle.load(open(pkl_file_path, "rb"))
            print(f"The activation data is loaded from {pkl_file_path}")
        else:
            raise FileNotFoundError(f"{pkl_file_path} does not exist!")
        
        self.cache_data_dict[lang] = data_dict
        
        return data_dict 
    
    def get_activity(self, data_dict: dict, fact_uuid: str, is_right_only: bool = False) -> Tuple[torch.tensor, int]:
        
        act = torch.stack(data_dict[fact_uuid][self.output_type], dim=0) # (L, d)
        rel = data_dict[fact_uuid]["rel_uri"]
        
        act_list = []
        for uuid, val in data_dict.items():
            if type(val) is not str:
                if is_right_only:
                    criteria = (val["rel_uri"] == rel) and (bool(val["is_match"]) == True)
                else:
                    criteria = (val["rel_uri"] == rel) 
                    
                if criteria:
                    act_list.append(torch.stack(val[self.output_type], dim=0)) # List[R x (L, d)]

        if len(act_list) <= 10:
            return None, len(act_list)
                  
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
    
    def plot_activity(self, data_dict: dict, lang: str, fact_uuid: str, save_dir: Path) -> torch.tensor:

        plot_path = Path(save_dir, f"{fact_uuid}_activity_heatmap.png")
        act_data, R = self.get_activity(data_dict=data_dict, fact_uuid=fact_uuid, is_right_only=True) # (L, d)
        if act_data is None:
            return
        
        binned_act = self.get_binned_activity(act_data=act_data) # (L, num_bins)
        L = binned_act.shape[0]
        
        plt.figure(figsize=(8, 10))
        plt.imshow(binned_act.cpu(), aspect='auto', cmap='viridis_r')
        plt.colorbar(label='Activation Value')
        plt.xlabel('Bins')
        plt.ylabel('Transformer Layers (0: bottom)')
        plt.yticks(ticks=range(L), labels=reversed(range(L)))
        
        obj = str(self.mlama_dataset.uuid_info_all_lang[fact_uuid]["en"]["obj"])
        pred = bool(data_dict[fact_uuid]["is_match"])
        
        example = self.mlama_dataset.uuid_info_all_lang[fact_uuid]["en"]["rel"].replace("[X]", 
            str(self.mlama_dataset.uuid_info_all_lang[fact_uuid]["en"]["sub"]))
        title = (f'Neuron Activity Pattern for Fact Translated to Lang: ' + str(Language.get(lang).display_name()) + 
                f'\nFact: {example} ([Y]: {obj}), Prediction: {pred}' + f"\nAvg over {R} triplets of same relation with only correct predictions")
        plt.title(title)
        plt.savefig(str(plot_path), dpi=300)
        plt.close()
    
        return binned_act

    def plot_activity_for_1_lang(self, lang: str, pred_type: bool, num_uuids: int = 10):
        
        if not self.cache_data_dict:
            data_dict = self.load_pkl_activation_file(lang=lang)
        else:
            data_dict = self.cache_data_dict[lang]

        save_dir = Path(self.heatmap_dir, lang, str(pred_type))
        save_dir.mkdir(exist_ok=True, parents=True)
        
        uuid_list = []
        for uuid, val in self.mlama_dataset.uuid_info_all_lang.items():
            if lang in val.keys():
                try:
                    if bool(data_dict[uuid]["is_match"]) == pred_type:
                        if "en" in self.mlama_dataset.uuid_info_all_lang[uuid].keys():
                            uuid_list.append(uuid)          
                except KeyError:
                    pass

        if num_uuids == -1:
            num_uuids = len(uuid_list)
            
        for fact_uuid in random.sample(uuid_list, k=min(num_uuids, len(uuid_list))):
            self.plot_activity(data_dict=data_dict, lang=lang, fact_uuid=fact_uuid, save_dir=save_dir)
    
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
            
    def get_ranking_by_probeless(self, label_list: List[str], layer_act_list: List[torch.tensor], num_rank: int = 50) -> torch.tensor:
            
        unique_labels = set(label_list)
        mean_act_dict = {}
        for label in unique_labels:
            index_list = [i for i, l in enumerate(label_list) if l == label]
            q = torch.stack([layer_act_list[i] for i in index_list], dim=0).mean(dim=0) # (Ni, L, d) -> (L, d)
            mean_act_dict[label] = {"q": q, "Ni": len(index_list)}
        
        sum_of_diff = []
        for z1, z2 in list(combinations(unique_labels, 2)):
            q1 = mean_act_dict[z1]["q"]
            q2 = mean_act_dict[z2]["q"]
            sum_of_diff.append(torch.abs(q1-q2))
            
        r = torch.stack(sum_of_diff).sum(dim=0) # (N, L, d) ->  (L, d)
        ranking = r.sort(dim=-1)[-1][:, :num_rank] # (L, num_rank)
        
        return ranking
    
    def measure_cross_lingual_fact_rep(self, lang1: str, lang2: str) -> List[float]:
        
        if lang1 in self.cache_data_dict.keys():
            data_dict1 = self.cache_data_dict[lang1]
        else:
            data_dict1 = self.load_pkl_activation_file(lang=lang1)
            
        if lang2 in self.cache_data_dict.keys():
            data_dict2 = self.cache_data_dict[lang2]
        else:
            data_dict2 = self.load_pkl_activation_file(lang=lang2)
        
        uuid_list = []
        layer_act_list1 = []
        layer_act_list2 = []
        label_list1 = []
        label_list2 = []
        c = 0
        
        for uuid, val in self.mlama_dataset.uuid_info_all_lang.items():
            if (lang1 in val.keys()) and (lang2 in val.keys()):
                if data_dict1[uuid]["is_match"] == data_dict2[uuid]["is_match"]:
                    act_data1 = self.get_activity(data_dict=data_dict1, fact_uuid=uuid) # (L, d)
                    act_data2 = self.get_activity(data_dict=data_dict2, fact_uuid=uuid) # (L, d)
                    
                    binned_act1 = self.get_binned_activity(act_data=act_data1) # (L, num_bins)
                    binned_act2 = self.get_binned_activity(act_data=act_data2) # (L, num_bins)
                    binned_act_diff = torch.abs(binned_act1 - binned_act2) # (L, num_bins)
                    
                    criteria1 = (binned_act_diff < 0.05).to(torch.float32).mean().item()
                    criteria2 = (binned_act_diff < 0.1).to(torch.float32).mean().item()
                    criteria3 = (binned_act_diff < 0.5).to(torch.float32).mean().item()
                    
                    if (criteria1 > 0.5) and (criteria2 > 0.75) and (criteria3 > 0.90):
                        print(c)
                        c += 1
                        uuid_list.append(uuid)
                        
                        label1 = data_dict1[uuid]["obj"]
                        layer_act_list1.append(act_data1) # List[N x (L, d)]
                        label_list1.append(label1)
                        
                        label2 = data_dict2[uuid]["obj"]
                        layer_act_list2.append(act_data2) # List[N x (L, d)]
                        label_list2.append(label2)
        
        rank1 = self.get_ranking_by_probeless(label_list=label_list1, layer_act_list=layer_act_list1) # (L, num_rank)
        rank2 = self.get_ranking_by_probeless(label_list=label_list2, layer_act_list=layer_act_list2) # (L, num_rank)
        
        jac_sim_list = []
        for i in range(rank1.shape[0]):
            jac_sim = len(set(rank1[i,:]).intersection(set(rank2[i,:]))) / len(set(rank1[i,:]).union(set(rank2[i,:])))
            jac_sim_list.append(jac_sim)
        
        return jac_sim_list
           
def main(model_name: str, device: torch.device):
       
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
    activation = Activation(device=device, mlama_dataset=mlama_dataset, tokenizer=tokenizer, model=model, name=name)
    
    for lang in ['zh', 'ja', 'ko', 'bn', 'hi']:
        activation.cache_data_dict = {}
        activation.plot_activity_for_1_lang(lang=lang, pred_type=True, num_uuids=20)
        activation.plot_activity_for_1_lang(lang=lang, pred_type=False, num_uuids=20)
    
if __name__ == "__main__":
    cuda_ids = [6]
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
    for model_name in models_list[:1]:
        main(model_name=model_name, device=device)