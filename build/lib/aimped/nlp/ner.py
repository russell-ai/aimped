# Author: AIMPED
# Date: 2023-March-11
# Description: NER model results

def NerModelResults(sents_tokens_list, sentences, tokenizer, model, text, device,
                    assertion_relation=False):
    """
    It returns the NER model results of a text.
    Parameters
    ----------
    sents_tokens_list : list
    tokenizer : transformers.PreTrainedTokenizer
    model : transformers.PreTrainedModel
    text : str
    device : torch.device
    assertion_relation : bool, optional
        The default is False.
    sentences : list, optional
        The default is []. Only used if assertion_relation is True

    Returns
    -------
    tokens : list
    preds : list
    probs : list
    begins : list
    ends : list
    sent_begins : list
        Only returned if assertion_relation is True
    sent_ends : list
        Only returned if assertion_relation is True
    sent_idxs : list
        Only returned if assertion_relation is True
    """
    import torch
    start = 0
    tokens, probs, begins, ends, preds, sent_begins, sent_ends, sent_idxs = [], [], [], [], [], [], [], []

    for sentence_idx, sent_token_list in enumerate(sents_tokens_list):
        start_sent = 0
        start = text.find(sentences[sentence_idx], start)
        model_inputs = tokenizer(sent_token_list, is_split_into_words=True, truncation=True,
                                 padding=False, max_length=512, return_tensors="pt").to(device)
        word_ids = model_inputs.word_ids()  # sub tokenlar sent_token_list deki hangi idxteki tokena ait
        # ornek word_ids = [None, 0, 1, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 11, 12, 13, 14, 15, None]
        outputs = model(**model_inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).tolist()[0]
        predictions = outputs.logits.argmax(dim=-1).tolist()[0]
        idx = 1
        while idx < len(word_ids) - 1:  # sondaki None icin islem yapmamak icin -1 yapildi
            word_id1 = word_ids[idx]
            word_id2 = word_ids[idx + 1]
            label = model.config.id2label[predictions[idx]]
            prob = max(probabilities[idx])
            if word_id1 == word_id2:
                while word_id1 == word_ids[idx]:
                    idx += 1
                idx -= 1

            token = sent_token_list[word_ids[idx]]
            begin = text.find(token, start)
            end = begin + len(token)
            tokens.append(token)
            begins.append(begin)
            ends.append(end)
            preds.append(label)
            probs.append(prob)
            start = end
            if assertion_relation:
                sentence_begin = sentences[sentence_idx].find(token, start_sent)
                sentence_end = sentence_begin + len(token)
                sent_begins.append(sentence_begin)
                sent_ends.append(sentence_end)
                sent_idxs.append(sentence_idx)
                start_sent = sentence_end
            idx += 1
    if not assertion_relation:
        return tokens, preds, probs, begins, ends
    else:
        return tokens, preds, probs, begins, ends, sent_begins, sent_ends, sent_idxs


# TODO : will be tested
if __name__ == "__main__":
    import torch
    import pandas as pd
    import numpy as np
    import os
    import re
    import json
    from tokenizer import SentenceTokenizer, WordTokenizer

    # load model
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    saved_model_path = r"C:\Users\rcali\Desktop\kubeflow\deid-ner-gitlab\model"
    tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
    model = AutoModelForTokenClassification.from_pretrained(saved_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    text = open("aimped\\test\\data.txt", "r").read()
    sents_tokens_list = WordTokenizer(SentenceTokenizer(text, "english"))

    # ner = pipeline('token-classification', model=model, tokenizer=tokenizer, device=device)
    # print(ner(text))

    # print(sents_tokens_list)
    tokens, preds, probs, begins, ends = NerModelResults(sents_tokens_list, tokenizer, model, text, device)

    white_label_list = ['PATIENT', 'ORGANIZATION', 'SSN', 'SEX', 'DOCTOR', 'HOSPITAL', 'AGE', 'MEDICALRECORD', 'ZIP',
                        'STREET', 'EMAIL', 'DATE', 'ID', 'CITY', 'COUNTRY', 'PROFESSION']

    from chunker import ChunkMerger

    merged_chunks = ChunkMerger(text, white_label_list, tokens, preds, probs, begins, ends)
    print(merged_chunks)
