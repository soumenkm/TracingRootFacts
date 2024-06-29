import torch, torchinfo, json, sys
from typing import List, Tuple
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from models import BertModelForProbing, XLMRobertaModelForProbing, get_mask_token
from transformers import BertTokenizer, XLMRobertaTokenizer
from mask_dataset import MaskedDataset

def prepare_dataset(tokenizer: "Tokenizer", mlama_dataset: MaskedDataset, lang1: "str", lang2: "str", num_of_samples: int) -> dict:
    
    mask_token = tokenizer.mask_token
    data_list1 = []
    data_columns = ["index", "uuid", "sent", "lang", "relid", "obj", "obj_uri", "sub", "sub_uri"]
    data_list2 = []

    index = 0
    for uuid, val in mlama_dataset.uuid_info_all_lang.items():
        if set([lang1, lang2]) <= (set(val.keys())):
            sent1 = val[lang1]["rel"].replace("[X]", val[lang1]["sub"])
            sent2 = val[lang2]["rel"].replace("[X]", val[lang2]["sub"])
            relid1 = val[lang1]["rel_uri"]
            relid2 = val[lang2]["rel_uri"]
            obj1 = val[lang1]["obj"]
            obj_uri1 = val[lang1]["obj_uri"]
            obj2 = val[lang2]["obj"]
            obj_uri2 = val[lang2]["obj_uri"]
            sub1 = val[lang1]["sub"]
            sub_uri1 = val[lang1]["sub_uri"]
            sub2 = val[lang2]["sub"]
            sub_uri2 = val[lang2]["sub_uri"]
            
            assert sub_uri1 == sub_uri2 and obj_uri1 == obj_uri2 and relid1 == relid2
            
            if tokenizer.tokenize(obj2).__len__() == 1:
                data_list1.append([index, uuid, sent1, lang1, relid1, obj1, obj_uri1, sub1, sub_uri1])
                data_list2.append([index, uuid, sent2, lang2, relid2, obj2, obj_uri2, sub2, sub_uri2])
                index += 1
    
    df1 = pd.DataFrame(data=data_list1, columns=data_columns)
    df2 = pd.DataFrame(data=data_list2, columns=data_columns)
    df1 = df1.sample(frac=1, random_state=42).reset_index(drop=True)
    df2 = df2.sample(frac=1, random_state=42).reset_index(drop=True)
    
    useful_df1 = df1.iloc[:num_of_samples, [2,5]]
    useful_df2 = df2.iloc[:num_of_samples, [2,5]]
    uuid_df = df1.iloc[:num_of_samples, 1]
    useful_df1["obj"] = useful_df1["obj"].apply(lambda x: tokenizer.tokenize(x))
    useful_df2["obj"] = useful_df2["obj"].apply(lambda x: tokenizer.tokenize(x))
    
    for i in range(useful_df1.shape[0]):
        new_mask = mask_token * len(useful_df1.iloc[i,1])
        useful_df1.iloc[i,0] = useful_df1.iloc[i,0].replace("[Y]", new_mask.strip())
    for i in range(useful_df2.shape[0]):
        new_mask = mask_token * len(useful_df2.iloc[i,1])
        useful_df2.iloc[i,0] = useful_df2.iloc[i,0].replace("[Y]", new_mask.strip())
        
    return {lang1: {"inputs_list": useful_df1["sent"].tolist(), "true_list": useful_df1["obj"].tolist()},
            lang2: {"inputs_list": useful_df2["sent"].tolist(), "true_list": useful_df2["obj"].tolist()},
            "uuid": uuid_df.tolist()}

def write_json_neuron_dataset(neuron_dataset: dict, model: "Model", tokenizer: "Tokenizer", lang1: str, lang2: str, model_name: str) -> None:
    
    inputs_list1 = neuron_dataset[lang1]["inputs_list"]
    true_list1 = neuron_dataset[lang1]["true_list"]
    inputs1 = tokenizer(inputs_list1, padding=True, truncation=True, return_tensors="pt")
    inputs_list2 = neuron_dataset[lang2]["inputs_list"]
    true_list2 = neuron_dataset[lang2]["true_list"]
    inputs2 = tokenizer(inputs_list2, padding=True, truncation=True, return_tensors="pt")
    uuid_list = neuron_dataset["uuid"]
    
    model.eval()
    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)
    
    mask_out1 = get_mask_token(tokenizer, outputs1)
    mask_out2 = get_mask_token(tokenizer, outputs2)

    out_dict1 = {}
    out_dict2 = {}
    for i, (item1, item2) in enumerate(zip(inputs_list1, inputs_list2)):
        if tokenizer.convert_tokens_to_ids(true_list1[i]) == tokenizer.convert_tokens_to_ids(mask_out1["token"][i]):
            if tokenizer.convert_tokens_to_ids(true_list2[i]) == tokenizer.convert_tokens_to_ids(mask_out2["token"][i]):
                out_dict1[i] = {"input": item1, "uuid": uuid_list[i], "mask_true_token": true_list1[i], "mask_pred_token": mask_out1["token"][i]}
                out_dict2[i] = {"input": item2, "uuid": uuid_list[i], "mask_true_token": true_list2[i], "mask_pred_token": mask_out2["token"][i]}
    
    json.dump(out_dict1, open(f"neuron_outputs/{lang1}_{model_name}_prediction.json","w"), ensure_ascii=False, indent=4)
    json.dump(out_dict2, open(f"neuron_outputs/{lang2}_{model_name}_prediction.json","w"), ensure_ascii=False, indent=4)

def get_binned_activation(model_outputs: dict) -> torch.tensor:

    ff_out_list = model_outputs["ff_out"] # List[L x (b, T, d)]
    ff_out_mod_list = []
    
    for ff_out in ff_out_list: 
        assert ff_out.shape[0] == 1
        item = ff_out.squeeze(0) # (T, d)
        index = model_outputs["mask_index"][0][0]
        item = item[index, :] # (d,)
        ff_out_mod_list.append(item)
    
    num_bins = 16
    num_layers = len(ff_out_mod_list)
    binned_activations = []
    for i, layer_activations in enumerate(ff_out_mod_list):
        d = layer_activations.shape[0]
        item = torch.split(layer_activations, d//num_bins, dim=0)
        binned_activations.append([i.mean().item() for i in item])
    
    binned_activations = torch.tensor(binned_activations)
    binned_activations = binned_activations - binned_activations.min() / (binned_activations.max() - binned_activations.min())
    return binned_activations

def plot_activation(binned_activations: torch.tensor, lang: str) -> None:
    
    plt.figure(figsize=(8, 6))
    plt.imshow(binned_activations, aspect='auto', cmap='viridis')
    plt.colorbar(label='Activation Value')
    plt.xlabel('Bins')
    plt.ylabel('Transformer Layers')
    plt.title(f'Binned Activations Across Transformer Layers for Lang: {lang}')
    plt.savefig(f'neuron_outputs/lang_{lang}_binned_activations_heatmap.png', dpi=300)

def get_neuron_activity(uuid: str, neuron_dataset: dict, model: "Model", tokenizer: "Tokenizer", lang1: str, lang2: str, model_name: str) -> None:
    
    index = neuron_dataset["uuid"].index(uuid)
    
    pre_inputs1 = neuron_dataset[lang1]["inputs_list"][index]
    true1 = neuron_dataset[lang1]["true_list"][index]
    inputs1 = tokenizer(pre_inputs1, padding=True, truncation=True, return_tensors="pt")
    pre_inputs2 = neuron_dataset[lang2]["inputs_list"][index]
    true2 = neuron_dataset[lang2]["true_list"][index]
    inputs2 = tokenizer(pre_inputs2, padding=True, truncation=True, return_tensors="pt")
    
    model.eval()
    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)
    
    ff_bin_out1 = get_binned_activation(model_outputs=outputs1)
    ff_bin_out2 = get_binned_activation(model_outputs=outputs2)
    
    plot_activation(ff_bin_out1, lang=lang1)
    plot_activation(ff_bin_out2, lang=lang2)
       
def main(model_name: str, lang1: str, lang2: str, num_of_samples: int=10):
       
    if "bert-" in model_name:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModelForProbing(tokenizer=tokenizer)
        is_mbert = True
    elif "xlm-roberta" in model_name:
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        model = XLMRobertaModelForProbing(tokenizer=tokenizer)
        is_mbert = False
    else:
        raise NotImplementedError("Invalid model name!")
    
    mlama_dataset = MaskedDataset()
    dataset = prepare_dataset(tokenizer=tokenizer, mlama_dataset=mlama_dataset, lang1=lang1, lang2=lang2, num_of_samples=num_of_samples)
    # write_json_neuron_dataset(neuron_dataset=dataset, model=model, tokenizer=tokenizer, lang1=lang1, lang2=lang2, model_name=model_name)
    uuid = "5a385050-7233-4e81-9776-c3226669ca8b"
    get_neuron_activity(uuid=uuid, neuron_dataset=dataset, model=model, tokenizer=tokenizer, lang1=lang1, lang2=lang2, model_name=model_name)
    
if __name__ == "__main__":
    models_list = ["bert-base-multilingual-cased", "xlm-roberta-large"]
    
    for model_name in models_list[1:]:
        main(model_name=model_name, lang1="en", lang2="bn", num_of_samples=-1)