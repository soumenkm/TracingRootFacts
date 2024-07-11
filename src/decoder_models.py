import torch, torchinfo, json, os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Tuple
import pandas as pd
from pathlib import Path

class LlamaModelForProbing(torch.nn.Module):
    
    def __init__(self, tokenizer: AutoTokenizer):
        
        super(LlamaModelForProbing, self).__init__()
        self.pretrain_llama_name = tokenizer.name_or_path
        self.model = AutoModelForCausalLM.from_pretrained(self.pretrain_llama_name)
        self.tokenizer = tokenizer
    
    def hook_function(self, module: torch.nn.Module, inputs: torch.tensor, outputs: torch.tensor):
        self.ff_out_neuron_output_list.append(outputs.detach())
    
    def register_hook(self):
        
        self.hooks_list = []
        self.ff_out_neuron_output_list = [] # List[L x (b, T, d)]
        for L in self.model.model.layers:
            h = L.mlp.register_forward_hook(self.hook_function)
            self.hooks_list.append(h)
    
    def remove_hook(self):
        
        for h in self.hooks_list:
            h.remove()
        self.hooks_list = []
        self.ff_out_neuron_output_list = []
            
    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor) -> dict:
        """input_ids = attention_mask = token_type_ids = (b, T)
        attention_mask = 0 means padded token and 1 means real token
        """
        act_output = [] # List[N x (L, d)]
        logits_output = [] # List[N x (d,)]
        
        self.register_hook()
        prediction_output = self.model(input_ids=input_ids, attention_mask=attention_mask).logits # (b, T, d)
        act_output.append(torch.stack([i[0, -1, :] for i in self.ff_out_neuron_output_list], dim=0))
        logits_output.append(prediction_output[0, -1])
        self.remove_hook()
            
        prompt_ids = input_ids # (1, T)
        for i in range(20):
            last_id = torch.tensor([[prediction_output.argmax(dim=-1)[0, -1]]]).to(input_ids.device) # (1, 1)
            last_attn_id = torch.tensor([[1]]).to(input_ids.device) # (1, 1)
            if "\n" in self.tokenizer.decode(last_id[0]):
                break
            else:
                input_ids = torch.cat([input_ids, last_id], dim=-1) # (1, T+1)
                attention_mask = torch.cat([attention_mask, last_attn_id], dim=-1) # (1, T+1)
                
                self.register_hook()
                prediction_output = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                act_output.append(torch.stack([i[0, -1, :] for i in self.ff_out_neuron_output_list], dim=0))
                logits_output.append(prediction_output[0, -1])
                self.remove_hook()
        
        assert logits_output.__len__() == act_output.__len__()
        
        if logits_output.__len__() > 1:
            return {"logits": torch.stack(logits_output[0:-1], dim=0), # (N, d)
                    "ff_out": torch.stack(act_output[0:-1], dim=0) # (N, L, d)
            } 
        else:
            return {"logits": torch.stack(logits_output, dim=0), # (N, d)
                    "ff_out": torch.stack(act_output, dim=0) # (N, L, d)
            } 
  
def prepare_dataset(tokenizer: "Tokenizer", csv_file_path: Path, num_of_samples: int, is_mlama: bool) -> dict:
    
    dataframe = pd.read_csv(csv_file_path)
    if is_mlama:
        dataframe = dataframe.sample(frac=1, random_state=42).reset_index(drop=True)
    
    mask_token = "[MASK] "
    useful_df = dataframe.loc[:num_of_samples, ["sent", "obj"]]

    for i in range(useful_df.shape[0]):
        useful_df.iloc[i,0] = useful_df.iloc[i,0].replace("[Y] ", mask_token)
    
    return {"inputs_list": useful_df["sent"].tolist(), 
            "true_list": useful_df["obj"].tolist()}
         
def inference(model_name: str, is_mlama: bool) -> None:
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = ("Predict the correct text ONLY at masked position.\n" + 
        "Input: {input}\nPredicted masked output:")
    
    if "llama" in model_name:
        model = LlamaModelForProbing(tokenizer=tokenizer).to("cuda")
        name = "llama3-7B-instruct"
    else:
        raise NotImplementedError("Invalid model name!")
        
    if is_mlama:
        data_file_path = Path("/raid/speech/soumen/MS_Research/TracingRootFacts/datasets/en.csv")
        dataset_name = "mlama_facts"
    else:
        data_file_path = Path("/raid/speech/soumen/MS_Research/TracingRootFacts/datasets/common.csv")
        dataset_name = "common_facts"
        
    dataset = prepare_dataset(tokenizer=tokenizer, csv_file_path=data_file_path, num_of_samples=20, is_mlama=is_mlama) 
    inputs_list = dataset["inputs_list"]
    true_list = dataset["true_list"]
    inputs = []
    for elem in inputs_list:
        inputs.append({k: v.to("cuda") for k, v in tokenizer([prompt.format(input=elem)], return_tensors="pt").items()})

    model.eval()
    with torch.no_grad():
        out = model(**(inputs[0]))

    # model.eval()
    # with torch.no_grad():
    #     outputs = []
    #     for i, elem in enumerate(inputs):
    #         gen_ids = model.generate(**elem, max_new_tokens=20)
    #         gen_ids_mod = []
    #         for g in gen_ids[0, len(elem["input_ids"][0]):]:
    #             if "\n" in tokenizer.decode(g):
    #                 break
    #             else:
    #                 gen_ids_mod.append(g)
            
    #         outputs.append({"input": inputs_list[i],
    #             "pred_output": tokenizer.batch_decode(gen_ids_mod, skip_special_tokens=True)[0].strip(),
    #             "true_output": true_list[i]})

    # json.dump(outputs, open(f"{name}_output.json", "w"), indent=4)
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    models = ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mixtral-8x7B-Instruct-v0.1"]
 
    inference(models[0], is_mlama=False)
    