import torch, torchinfo, json
from transformers import BertForMaskedLM, BertTokenizer
from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput
from typing import List, Tuple

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
    
    def get_mask_token(self, predictions_output: dict) -> dict:
        """predictions_output = (b, T, d)"""
        
        model_logit_output = predictions_output["logits"].argmax(dim=-1) # (b, T)
        predicted_token_output = []
        predicted_token_id_output = []

        for i, mask_index in enumerate(predictions_output["mask_index"]):
            token_ids = model_logit_output[i, mask_index].tolist()
            predicted_token_output.append([self.tokenizer.decode([t]) for t in token_ids])
            predicted_token_id_output.append(token_ids)
        
        return  {"token_ids": predicted_token_id_output, "token": predicted_token_output}
            
def main(model_name: str) -> None:
        
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModelForProbing(tokenizer=tokenizer)
    inputs_list = [
        "The national flag of [MASK] is known as Tiranga.",
        "The prime minister of India since 2014 is [MASK] [MASK] [MASK] [MASK] [MASK].",
        "The capital of India is [MASK] [MASK].",
        "The Indian festival of lights is called [MASK] [MASK].",
        "The national bird of India is [MASK] [MASK].",
        "The Indian independence day is celebrated on [MASK] [MASK].",
        "The currency of India is the [MASK] [MASK] [MASK].",
        "The Indian city known as the 'City of Joy' is [MASK].",
        "The river considered holy in India is the [MASK] [MASK].",
        "The largest state in India by population is [MASK] [MASK].",
        "The president of the United States lives in the [MASK] house.",
        "The capital of the United States is [MASK] D.C.",
        "The American city known as the 'Silicon Valley' is [MASK].",
        "The river that runs through the Grand Canyon is the [MASK] river.",
        "The largest state in the United States by area is [MASK].",
        "The American space agency is called [MASK]."
    ]
    
    inputs = tokenizer(inputs_list, padding=True, truncation=True, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    mask_out = model.get_mask_token(outputs)

    out_dict = {}
    for i, item in enumerate(inputs_list):
        out_dict[i] = {"input": item, "mask_pred_token": mask_out["token"][i]}
    
    json.dump(out_dict, open(f"outputs/{model_name}_prediction.json","w"), indent=4)
    
if __name__ == "__main__":
    
    models = ["bert-base-cased", "bert-base-multilingual-cased", "bert-large-cased", "bert-large-cased-whole-word-masking"]

    for name in models:
        main(name)