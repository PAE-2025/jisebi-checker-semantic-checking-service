import difflib
from transformers import AutoTokenizer, T5ForConditionalGeneration
import transformers.utils
from typing import Dict, List, Union
import torch

transformers.utils.move_cache()

# Inisialisasi model dan tokenizer Grammarly
tokenizer = AutoTokenizer.from_pretrained("vennify/t5-base-grammar-correction")
model = T5ForConditionalGeneration.from_pretrained("vennify/t5-base-grammar-correction", low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
model.eval()

async def process_text(text: Union[str, List[str]]) -> Dict[str, any]:

    
    try:

        if isinstance(text, str):
            # Prompt hanya untuk memperbaiki grammar
            prompt = "Fix grammar and typos: "
            input_text = f"{prompt}{text}"

            input_ids = tokenizer(input_text, return_tensors="pt").input_ids
            outputs = model.generate(input_ids, max_length=256, num_beams=4, early_stopping=True)
            corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Bandingkan kata demi kata untuk mendeteksi typo
            highlighted_typos = compare_words(text.strip(), corrected_text)
            
            return {
                "original": text,
                "corrected": corrected_text,
                "highlighted_typos": highlighted_typos
            }
        
        elif isinstance(text, list):
            results = []
            for element in text:
                # Prompt hanya untuk memperbaiki grammar
                prompt = "Fix grammar and typos: "
                input_text = f"{prompt}{element}"

                input_ids = tokenizer(input_text, return_tensors="pt").input_ids
                outputs = model.generate(input_ids, max_length=256, num_beams=4, early_stopping=True)
                corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Bandingkan kata demi kata untuk mendeteksi typo
                highlighted_typos = compare_words(element.strip(), corrected_text)

                results.append(
                    {
                        "original": element,
                        "corrected": corrected_text,
                        "highlighted_typos": highlighted_typos
                    }
                )

            return results


    except Exception as e:
        raise Exception(f"Error processing text: {str(e)}")

def compare_words(original: str, corrected: str) -> List[Dict[str, str]]:
    original_words = original.split()
    corrected_words = corrected.split()
    
    # Gunakan difflib untuk membandingkan kata
    matcher = difflib.SequenceMatcher(None, original_words, corrected_words)
    highlighted_typos = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ['replace', 'delete']:  # Kata yang dianggap typo
            for i in range(i1, i2):
                highlighted_typos.append({
                    "word": f"_{original_words[i]}_",  # Tambahkan underscore untuk typo
                    "status": "typo (underlined)"
                })
        elif tag == 'equal':  # Kata yang benar
            for i in range(i1, i2):
                highlighted_typos.append({
                    "word": original_words[i],
                    "status": "correct"
                })
        elif tag == 'insert':  # Kata yang ditambahkan di corrected text
            for j in range(j1, j2):
                highlighted_typos.append({
                    "word": corrected_words[j],
                    "status": "suggested addition"
                })
    
    return highlighted_typos