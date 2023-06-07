#Author AIMPED
#Date 2023-March-20
#Description Contains the model loading functions for aimped library.

# write model loading functions here

def load_model(model_path, task):
    """
    Loads the model and tokenizer from the model path.
    params:
        model_path: path to the model
        task: task of the model
    returns:
        model: the model
        tokenizer: the tokenizer
    """
    if task == "token_classification":
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        return model, tokenizer
    elif task == "question_answering":
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        return model, tokenizer
    elif task == "Classification":
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        return model, tokenizer
    elif task == "Translation":
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        return model, tokenizer
    elif task == "Summarization":
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        return model, tokenizer
    elif task == "Text Generation":
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        return model, tokenizer


def save_model(model, tokenizer, model_path):
    """
    Saves the model and tokenizer to the model_path.
    params:
        model: the model
        tokenizer: the tokenizer
        model_path: path to the model
    """
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
