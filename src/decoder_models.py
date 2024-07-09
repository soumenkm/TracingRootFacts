import torch, torchinfo, json, os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Tuple
import pandas as pd
from pathlib import Path

class LlamaModelForProbing(torch.nn.Module):
    
    def __init__(self, tokenizer: AutoTokenizer):
        
        super(LlamaModelForProbing, self).__init__()
        self.pretrain_llama_name = tokenizer.name_or_path
        self.pretrain_llama_model = LlamaForCausalLM.from_pretrained(self.pretrain_llama_name)
        self.tokenizer = tokenizer
        
    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor, token_type_ids: torch.tensor) -> dict:
        """input_ids = attention_mask = token_type_ids = (b, T)
        attention_mask = 0 means padded token and 1 means real token
        """
        
        emb_out = self.pretrain_bert_model.bert.embeddings(input_ids=input_ids, token_type_ids=token_type_ids) # (b, T, d)
        extended_attention_mask = self.pretrain_bert_model.bert.get_extended_attention_mask(attention_mask=attention_mask, input_shape=emb_out.shape) # (b, 1, 1, T) - later this to be converted to (b, h, T, T)
        transformer_output = emb_out
        ff_act_neuron_output_list = []
        ff_out_neuron_output_list = []
        
        for transformer in self.pretrain_bert_model.bert.encoder.layer:

            attention_output = transformer.attention(hidden_states=transformer_output, attention_mask=extended_attention_mask)[0] # (b, T, d) - attention output is taken, not the attention weights
            ff_act_neuron_output = transformer.intermediate(hidden_states=attention_output) # (b, T, 4d)
            ff_out_neuron_output = transformer.output.dense(ff_act_neuron_output) # (b, T, d)
            hidden_states = transformer.output.dropout(ff_out_neuron_output) # (b, T, d)
            transformer_output = transformer.output.LayerNorm(hidden_states + attention_output) # (b, T, d)

            ff_act_neuron_output_list.append(ff_act_neuron_output)
            ff_out_neuron_output_list.append(ff_out_neuron_output)
            
        encoder_output = transformer_output # (b, T, d)
        prediction_output = self.pretrain_bert_model.cls(encoder_output) # (b, T, d) - logits
        
        is_mask_token = input_ids == self.tokenizer.mask_token_id # (b, T)
        mask_index_list = [None] * is_mask_token.shape[0] # (b,) -> List of list
        mask_index_tensor = is_mask_token.nonzero() # (z, 2) - Total number of masks = z
        
        for i in range(len(mask_index_list)):
            mask_index_list[i] = [j[1].item() for j in mask_index_tensor if j[0].item() == i]
            
        return {"logits": prediction_output, # (b, T, d)
                "ff_act": ff_act_neuron_output_list, # List[L x (b, T, 4d)]
                "ff_out": ff_out_neuron_output_list, # List[L x (b, T, d)]
                "mask_index": mask_index_list # List[b x List[z_b]]
            } 
    
class XGLMModelForProbing(torch.nn.Module):
    
    def __init__(self, tokenizer: AutoTokenizer):
        
        super(XGLMModelForProbing, self).__init__()
        self.pretrain_xglm_name = tokenizer.name_or_path
        self.pretrain_xglm_model = XGLMForCausalLM.from_pretrained(self.pretrain_xglm_name)
        self.tokenizer = tokenizer
        
    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor) -> dict:
        """input_ids = attention_mask = (b, T)
        attention_mask = 0 means padded token and 1 means real token
        """
        
        pass

def get_mask_token(tokenizer: "tokenizer", predictions_output: dict) -> dict:
    """predictions_output = (b, T, d)"""
    
    model_logit_output = predictions_output["logits"].argmax(dim=-1) # (b, T)
    predicted_token_output = []
    predicted_token_id_output = []

    for i, mask_index in enumerate(predictions_output["mask_index"]):
        token_ids = model_logit_output[i, mask_index].tolist()
        predicted_token_output.append(tokenizer.convert_ids_to_tokens(token_ids))
        predicted_token_id_output.append(token_ids)
    
    return  {"token_ids": predicted_token_id_output, "token": predicted_token_output}
    
def prepare_dataset(tokenizer: "Tokenizer", csv_file_path: Path, num_of_samples: int, is_mlama: bool, is_mbert: bool) -> dict:
    
    dataframe = pd.read_csv(csv_file_path)
    if is_mlama:
        dataframe = dataframe.sample(frac=1, random_state=42).reset_index(drop=True)
    
    if is_mbert:
        mask_token = "[MASK] "
    else:
        mask_token = "<mask> "
    
    useful_df = dataframe.loc[:num_of_samples, ["sent", "obj"]]
    
    useful_df["obj"] = useful_df["obj"].apply(lambda x: tokenizer.tokenize(x))
    for i in range(useful_df.shape[0]):
        new_mask = mask_token * len(useful_df.iloc[i,1])
        useful_df.iloc[i,0] = useful_df.iloc[i,0].replace("[Y] ", new_mask)
    
    return {"inputs_list": useful_df["sent"].tolist(), "true_list": useful_df["obj"].tolist()}
         
def main(model_name: str, is_mlama: bool) -> None:
    
    if "llama" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = LlamaModelForProbing(tokenizer=tokenizer)
    elif "xglm" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = XGLMModelForProbing(tokenizer=tokenizer)
    else:
        raise NotImplementedError("Invalid model name!")
        
    if is_mlama:
        data_file_path = Path("/raid/speech/soumen/MS_Research/TracingRootFacts/datasets/en.csv")
        dataset_name = "mlama_facts"
    else:
        data_file_path = Path("/raid/speech/soumen/MS_Research/TracingRootFacts/datasets/common.csv")
        dataset_name = "common_facts"
        
    dataset = prepare_dataset(tokenizer=tokenizer, csv_file_path=data_file_path, num_of_samples=20, is_mlama=is_mlama, is_mbert=is_mbert) 
    inputs_list = dataset["inputs_list"]
    true_list = dataset["true_list"]
    inputs = tokenizer(inputs_list, padding=True, truncation=True, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    mask_out = get_mask_token(tokenizer, outputs)

    out_dict = {}
    for i, item in enumerate(inputs_list):
        out_dict[i] = {"input": item, "mask_true_token": true_list[i], "mask_pred_token": mask_out["token"][i]}
    
    json.dump(out_dict, open(f"outputs/{dataset_name}_{model_name}_prediction.json","w"), indent=4)
       
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    models = ["facebook/xglm-564M", "meta-llama/Llama-2-7b-hf", "mistralai/Mixtral-8x7B-v0.1"]
    model_id = 1
    tokenizer = AutoTokenizer.from_pretrained(models[model_id])
    if model_id == 0:
        model = AutoModelForCausalLM.from_pretrained(models[model_id]).to("cuda")
        name = "xglm-564M"
    elif model_id == 1:
        model = AutoModelForCausalLM.from_pretrained(models[model_id]).to("cuda")
        name = "llama2-7B"
    elif model_id == 2:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(models[model_id], quantization_config=quantization_config)
        name = "mixtral8x7B-Q4"
    prompt_list = [
        "The capital of the USA is ",
        "The capital of the USA is <mask>",
        "The capital of the <mask> is Washington, D.C.",
        "Input: 'The capital of the USA is <mask>.'\nMasked output: ",
        "Predict the masked output.\nInput: 'The capital of the USA is <mask>.'\nMasked output: ",
        "Input: 'The capital of the <mask> is Washington, D.C.'\nMasked output: ",
        "Predict the masked output.\nInput: 'The capital of the <mask> is Washington, D.C.'\nMasked output: ",    
    ] 
    output_dict = {}   
    for i, prompt in enumerate(prompt_list):
    
        inputs = {k: v.to("cuda") for k, v in tokenizer([prompt], return_tensors="pt").items()}
        gen_ids = model.generate(**inputs, max_new_tokens=20)
        outputs = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]

        output_dict[i] = {"prompt": prompt, "output": outputs}
    
    json.dump(output_dict, open(f"{name}_output.json", "w"), indent=4)
    