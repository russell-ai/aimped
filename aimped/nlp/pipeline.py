# Author AIMPED
# Date 2023-March-11
# Description: This file contains the pipeline wrapper for NER, Assertion and De-identification models


task_list = [
    "ner",
    "assertion",
    "deid",

]


def pipeline(text, model, tokenizer,task="ner"):
    """
    It returns the results of a text.
    parameters:
    ----------------
    task: str
    text: str
    model: transformers.modeling_utils.PreTrainedModel
    tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizer
    return:
    ----------------
    results: list of dict
    """
    import pandas as pd
    import numpy as np
    from aimped.nlp import sentence_tokenizer, word_tokenizer
    if task == "ner":
        from aimped.nlp import ner_pipeline
        results = ner_pipeline(text, model, tokenizer)
        return results
    elif task == "assertion":
        from aimped.nlp import assertion_pipeline
        results = assertion_pipeline(text, model, tokenizer)
        return results
    elif task == "deid":
        from aimped.nlp import deid_pipeline
        results = deid_pipeline(text, model, tokenizer)
        return results
    else:
        print(f"Task {task} is not supported. Please choose from {task_list}")

