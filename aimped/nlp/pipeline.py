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

    def ner_result(self, text, sents_tokens_list, sentences, assertion_relation=False):
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
        ner_results = NerModelResults(text=text,
                                      tokenizer=self.tokenizer,
                                      model=self.model,
                                      device=self.device,
                                      sents_tokens_list=sents_tokens_list,
                                      sentences=sentences,
                                      assertion_relation=assertion_relation
                                      )

        return ner_results

    def deid_result(self, text, merged_results, fake_csv_path, faked=False, masked=False):
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

    def assertion_result(self, ner_results, sentences, classifier, assertion_white_label_list):
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

    def chunker_result(self, text, white_label_list, tokens, preds, probs, begins, ends,
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

    def regex_model_output_merger(self, regex_json_files_path, model_results, text, white_label_list):
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

    def relation_result(self, sentences, ner_chunk_results, relation_classifier,
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
        results = RelationResults(sentences=sentences,
                                  ner_chunk_results=ner_chunk_results,
                                  relation_classifier=relation_classifier,
                                  ner_white_label_list=ner_white_label_list,
                                  relation_white_label_list=relation_white_label_list,
                                  one_to_many=one_to_many,
                                  one_label=one_label,
                                  return_svg=return_svg)
        return results

 
   

    def __str__(self) -> str:
        """Return the string representation of the pipeline."""
        return f"Pipeline(model={self.model}, tokenizer={self.tokenizer})"


    def __str__(self) -> str:
        """Return the string representation of the pipeline."""
        return f"Pipeline(model={self.model}, tokenizer={self.tokenizer})"

