# Author: Aimped
# Date: 2023-March-11
# Description: This file contains model loading functions

def load_config(file_path='config.json'):
    with open(file_path, "r") as f:
        config = json.load(f)
    return config

def load_model(model_name):
    config = load_config()
    model_path = config['model_path']
    model = load_model(model_path + model_name)
    return model