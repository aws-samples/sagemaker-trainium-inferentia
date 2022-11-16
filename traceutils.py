import transformers
import torch
from pathlib import Path

def trace(sample_seq,checkpoint_path):
    
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    model = transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", return_dict=False)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    base_path = "traced_model/"
    Path(base_path).mkdir(exist_ok=True)
    save_path = base_path + "model.pth" 
   
    max_length = 128

    tokenized_sequence_pair = tokenizer.encode_plus(sample_seq, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")

    example = tokenized_sequence_pair["input_ids"], tokenized_sequence_pair["attention_mask"]

    traced_model = torch.jit.trace(model.eval(), example)
    traced_model.save(save_path)
    
    return save_path