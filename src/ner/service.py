# src/ner/service.py
from transformers import pipeline

ner_model = pipeline("ner")

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
