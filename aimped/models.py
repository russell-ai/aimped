# Author: AIMPED
# Date: 2023-March-11
# Description: This file contains model loading functions

def load_config(file_path='config.json'):
    import json
    with open(file_path, "r") as f:
        config = json.load(f)
    return config

def load_model(model_name):
    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification
    except ImportError:
        import sys
        sys.exit("Please install transformers library using 'pip install transformers' command")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    return tokenizer, model