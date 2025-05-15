# src/ner/service.py
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

model_name = "dslim/bert-base-NER"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16)
model.eval()

ner_model = pipeline("ner", model=model, tokenizer=tokenizer)

def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if merged_tokens and token['entity'].startswith('I-') and merged_tokens[-1]['entity'].endswith(token['entity'][2:]):
            last_token = merged_tokens[-1]
            last_token['word'] += token['word'].replace('##', '')
            last_token['end'] = token['end']
            last_token['score'] = float((last_token['score'] + token['score']) / 2)
        else:
            token['score'] = float(token['score'])
            merged_tokens.append(token)
    return merged_tokens

def run_ner(text: str):
    output = ner_model(text)
    for entity in output:
        entity["score"] = float(entity["score"])
    return merge_tokens(output)
