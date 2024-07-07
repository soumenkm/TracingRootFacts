import torch, torchinfo, json
from transformers import XLMRobertaTokenizer, XLMRobertaForMaskedLM
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaIntermediate, XLMRobertaOutput
from transformers import BertForMaskedLM, BertTokenizer
from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput
from typing import List, Tuple
import pandas as pd
from pathlib import Path

class BertModelForProbing(torch.nn.Module):
    
    def __init__(self, tokenizer: BertTokenizer):
        
        super(BertModelForProbing, self).__init__()
        self.pretrain_bert_name = tokenizer.name_or_path
        self.pretrain_bert_model = BertForMaskedLM.from_pretrained(self.pretrain_bert_name)
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
    
class XLMRobertaModelForProbing(torch.nn.Module):
    
    def __init__(self, tokenizer: XLMRobertaTokenizer):
        
        super(XLMRobertaModelForProbing, self).__init__()
        self.pretrain_xlmr_name = tokenizer.name_or_path
        self.pretrain_xlmr_model = XLMRobertaForMaskedLM.from_pretrained(self.pretrain_xlmr_name)
        self.tokenizer = tokenizer
        
    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor) -> dict:
        """input_ids = attention_mask = (b, T)
        attention_mask = 0 means padded token and 1 means real token
        """
        
        emb_out = self.pretrain_xlmr_model.roberta.embeddings(input_ids=input_ids) # (b, T, d)
        extended_attention_mask = self.pretrain_xlmr_model.roberta.get_extended_attention_mask(attention_mask=attention_mask, input_shape=emb_out.shape) # (b, 1, 1, T) - later this to be converted to (b, h, T, T)
        transformer_output = emb_out
        ff_act_neuron_output_list = []
        ff_out_neuron_output_list = []
        
        for transformer in self.pretrain_xlmr_model.roberta.encoder.layer:

            attention_output = transformer.attention(hidden_states=transformer_output, attention_mask=extended_attention_mask)[0] # (b, T, d) - attention output is taken, not the attention weights
            ff_act_neuron_output = transformer.intermediate(hidden_states=attention_output) # (b, T, 4d)
            ff_out_neuron_output = transformer.output.dense(ff_act_neuron_output) # (b, T, d)
            hidden_states = transformer.output.dropout(ff_out_neuron_output) # (b, T, d)
            transformer_output = transformer.output.LayerNorm(hidden_states + attention_output) # (b, T, d)

            ff_act_neuron_output_list.append(ff_act_neuron_output)
            ff_out_neuron_output_list.append(ff_out_neuron_output)
            
        encoder_output = transformer_output # (b, T, d)
        prediction_output = self.pretrain_xlmr_model.lm_head(encoder_output) # (b, T, d) - logits
        
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
    
    models = ["bert-base-multilingual-cased", "bert-large-cased-whole-word-masking", "xlm-roberta-large"]
    
    for model_name in models:
        main(model_name=model_name, is_mlama=True)
    for model_name in models:
        main(model_name=model_name, is_mlama=False)