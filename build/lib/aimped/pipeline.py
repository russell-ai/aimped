
def pipeline(task, tokenizer, model, text):
    input = tokenizer(text)
    result = model.predict(input)
    return result