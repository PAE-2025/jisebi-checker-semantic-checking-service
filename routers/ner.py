from fastapi import APIRouter
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

router = APIRouter()

model_name = "dslim/bert-base-NER"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

ner_model = pipeline("ner", model=model, tokenizer=tokenizer)
text = "Microsoft is headquartered in Redmond."

def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if merged_tokens and token['entity'].startswith('I-') and merged_tokens[-1]['entity'].endswith(token['entity'][2:]):
            last_token = merged_tokens[-1]
            last_token['word'] += token['word'].replace('##', '')
            last_token['end'] = token['end']
            last_token['score'] = float((last_token['score'] + token['score']) / 2)  # Ubah ke float
        else:
            token['score'] = float(token['score'])  # Ubah ke float
            merged_tokens.append(token)
    return merged_tokens

@router.post("/ner")
async def recognize_entities(text: str):
    output = ner_model(text)
    
    # Konversi nilai score dari numpy.float32 ke float
    for entity in output:
        entity["score"] = float(entity["score"])
    
    merged_output = merge_tokens(output)
    
    return {"text": text, "entities": merged_output}
