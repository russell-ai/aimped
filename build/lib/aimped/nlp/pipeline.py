# Author AIMPED
# Date 2023-March-11
# Description: This file contains the pipeline wrapper for NER, Assertion and De-identification models.


class Pipeline:
    """
    It returns the ner results of a text.
    parameters:
    ----------------
    task: str
    text: str
    model: transformers.modeling_utils.PreTrainedModel
    tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizer
    *args: arguments
    **kwargs: keyword arguments
    return:
    ----------------
    results: list of dict
    """

    def __init__(self, tokenizer, model, device='cpu'):
        """Initialize the pipeline class."""
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def ner_results(self, text, sents_tokens_list, sentences, assertion_relation=False):
        """It returns the ner results of a text.
        parameters:
        ----------------
        text: str
        sents_tokens_list: list of list of str
        assertion_relation: bool
        sentences: list of str
        return:
        ----------------
        ner_results: list of dict"""
        from aimped.nlp.ner import NerModelResults
        ner_results = NerModelResults(text=text, tokenizer=self.tokenizer,
                                      model=self.model, device=self.device,
                                      sents_tokens_list=sents_tokens_list,
                                      sentences=sentences,
                                      assertion_relation=assertion_relation
                                      )

        return ner_results

    def deid(self, text, merged_results, fake_csv_path, faked=False, masked=False):
        """It returns the deid results of a text.
        parameters:
        ----------------
        text: str
        merged_results: list of dict
        fake_csv_path: str
        faked: bool
        masked: bool
        return:
        ----------------
        results: list of dict
        """
        from aimped.nlp.deid import maskText, fakedChunk, fakedText, deidentification
        results = deidentification(text=text,
                                   merged_results=merged_results,
                                   fake_csv_path=fake_csv_path,
                                   faked=faked,
                                   masked=masked)
        return results

    def assertion(self, ner_results, sentences, classifier, assertion_white_label_list):
        """It returns the assertion results of a text.
        parameters:
        ----------------
        text: str
        ner_results: list of dict
        sentences: list of str
        classifier: str
        return:
        ----------------
        results: list of dict
        """
        from aimped.nlp.assertion import AssertionAnnotateSentence, AssertionModelResults
        results = AssertionModelResults(ner_results=ner_results,
                                        sentences=sentences,
                                        classifier=classifier,
                                        assertion_white_label_list=assertion_white_label_list
                                        )
        return results

    def chunk_merger(self, text, white_label_list, tokens, preds, probs, begins, ends,
                     assertion_relation=False, sent_begins=[], sent_ends=[], sent_idxs=[]):
        """It returns the merged chunks of a text.
        parameters:
        ----------------
        text: str
        white_label_list: list of str
        tokens: list of str
        preds: list of str
        probs: list of float
        begins: list of int
        ends: list of int
        assertion_relation: bool
        sent_begins: list of int
        sent_ends: list of int
        sent_idxs: list of int
        return:
        ----------------
        results: list of dict
        """
        from aimped.nlp.chunker import ChunkMerger
        results = ChunkMerger(text=text,
                              white_label_list=white_label_list,
                              tokens=tokens,
                              preds=preds,
                              probs=probs,
                              begins=begins,
                              ends=ends,
                              assertion_relation=assertion_relation,
                              sent_begins=sent_begins,
                              sent_ends=sent_ends,
                              sent_idxs=sent_idxs)
        return results

    def regex(self, regex_json_files_path, model_results, text, white_label_list):
        """It returns the regex results of a text.
        parameters:
        ----------------
        regex_json_files_path: str
        model_results: list of dict
        text: str
        white_label_list: list of str
        return:
        ----------------
        results: list of dict
        """
        from aimped.nlp.regex_parser import RegexNerParser, RegexModelNerMerger, RegexModelOutputMerger
        import glob
        regex_json_files_path_list = glob.glob(f"{regex_json_files_path}/*.json")
        merged_results = RegexModelOutputMerger(regex_json_files_path_list=regex_json_files_path_list,
                                                model_results=model_results,
                                                text=text,
                                                white_label_list=white_label_list)
        return merged_results

    def relation_results(self, sentences, ner_chunk_results, relation_classifier,
                         ner_white_label_list, relation_white_label_list, one_to_many=True,
                         one_label=None, return_svg=False):
        """It returns the relation results of a text.
        parameters:
        ----------------
        sentences: list of str
        ner_chunk_results: list of dict
        relation_classifier: str
        ner_white_label_list: list of str
        relation_white_label_list: list of str
        one_to_many: bool = True
        one_label: str = None
        return_svg: bool = False
        return:
        ----------------
        results: list of dict
        """
        from aimped.nlp.relation import RelationResults, RelationAnnotateSentence
        results = RelationResults(sentences, ner_chunk_results, relation_classifier,
                                  ner_white_label_list, relation_white_label_list, one_to_many=one_to_many,
                                  one_label=one_label, return_svg=return_svg)
        return results

    def __str__(self) -> str:
        """Return the string representation of the pipeline."""
        return f"Pipeline(model={self.model}, tokenizer={self.tokenizer})"


# TEST CODE
if __name__ == "__main__":
    import torch
    import pandas as pd
    import numpy as np
    import os
    import re
    import json
    from tokenizer import sentence_tokenizer, word_tokenizer

    # load model
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    saved_model_path = r"C:\Users\rcali\Desktop\kubeflow\deid-ner-gitlab\model"
    tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
    model = AutoModelForTokenClassification.from_pretrained(saved_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"device: {device}")

    # load data
    text = open("aimped\\test\\data.txt", "r").read()
    sentences = sentence_tokenizer(text, "english")
    sents_tokens_list = word_tokenizer(sentences)
    white_label_list = ['PATIENT', 'ORGANIZATION', 'SSN', 'SEX', 'DOCTOR', 'HOSPITAL', 'AGE', 'MEDICALRECORD', 'ZIP',
                        'STREET', 'EMAIL', 'DATE', 'ID', 'CITY', 'COUNTRY', 'PROFESSION']
    # print(sents_tokens_list)

    # test pipeline
    pipe = Pipeline(model=model, tokenizer=tokenizer, device='cpu')
    tokens, preds, probs, begins, ends = pipe.ner_results(text=text,
                                                          sents_tokens_list=sents_tokens_list,
                                                          sentences=sentences)
    # print("tokens: ", tokens)
    # print("preds: ", preds)
    # print("probs:", probs)
    # print("begins:", begins)
    # print("ends:", ends)
    merged_chunks = pipe.chunk_merger(text, white_label_list, tokens, preds, probs, begins, ends)
    # print(merged_chunks)

    # test regex
    regex_json_files_path = r"C:\Users\rcali\Desktop\kubeflow\deid-ner-gitlab\nlp-health-deidentification-sub-base-en\json_regex"
    merged_results = pipe.regex(regex_json_files_path, merged_chunks, text, white_label_list)
    # print(merged_results)

    # test deid
    fake_csv_path = r"C:\Users\rcali\Desktop\kubeflow\deid-ner-gitlab\nlp-health-deidentification-sub-base-en\fake.csv"
    deid_results = pipe.deid(text, merged_results, fake_csv_path, faked=True, masked=True)
    import pprint

    pprint.pprint(deid_results, indent=1)

    # test assertion
