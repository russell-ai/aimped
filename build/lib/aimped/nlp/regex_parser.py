# Author: AIMPED
# Date: 2023-March-12
# Description: This file contains the regex parser for de-identification of clinical notes

def RegexNerParser(path, text, white_label_list):
    """
    Finds all the chunks that correspond to the regex pattern, 
    and checks their prefix and suffix collocations in the scope of context length.
    parameters:
    path: str
    text: str
    return:
    parser_results: list of dict
    """
    import json
    import re

    with open(path, 'r', encoding="utf8") as f:
        file = json.load(f)
    label = file["label"]
    parser_results = []
    if label in white_label_list:
        chunks = re.findall(file["regex"], text)

        for chunk in set(chunks):
            seek = 0
            for i in range(chunks.count(chunk)):
                strt = text.find(chunk, seek) - file["contextLength"]
                if strt < 0:
                    strt = 0
                before = text[strt:text.find(chunk, seek)]
                after = text[
                        text.find(chunk, seek) + len(chunk):text.find(chunk, seek) + len(chunk) + file["contextLength"]]
                begin = text.find(chunk, seek)
                end = text.find(chunk, seek) + len(chunk)
                seek = end
                left_context, right_context = [], []
                if file["prefix"]:
                    left_context = [True if pre.lower() in before.lower() else False for pre in file["prefix"]]

                if file["suffix"]:
                    right_context = [True if sfx.lower() in after.lower() else False for sfx in file["suffix"]]
                if any(left_context) or any(right_context):
                    parser_results.append(
                        {'chunk': chunk, 'confidence': 1, 'begin': begin, 'end': end, 'entity': file["label"]})
                elif (not bool(file["prefix"])) and (not bool(file["suffix"])):
                    parser_results.append(
                        {'chunk': chunk, 'confidence': 1, 'begin': begin, 'end': end, 'entity': file["label"]})
                else:
                    # parser_results.append({'chunk': chunk, 'confidence':0.67, 'begin': begin, 'end':end, 'entity':file["label"]})
                    left_context, right_context = [], []
                    continue

    return parser_results


def RegexModelNerMerger(rule, results_from_model):
    """
    Merges the results from regex and model.
    parameters:
    rule: list of dict
    results_from_model: list of dict
    return:
    merged: list of dict
    """
    import operator
    merged = []
    out = []
    if results_from_model and rule:
        out = [j for j in results_from_model for z in rule if
               j['begin'] <= z['begin'] < j['end'] or j['begin'] < z['end'] <= j['end'] or z['begin'] <= j['begin'] < z[
                   'end'] or z['begin'] < j['end'] <= z['end']]
        for removed in out:
            if removed in results_from_model:
                results_from_model.remove(removed)
        merged += results_from_model + rule
    else:
        merged += results_from_model + rule
    merged.sort(key=operator.itemgetter('begin'))
    return merged


def RegexModelOutputMerger(regex_json_files_path_list, model_results, text, white_label_list):
    """Parses the text with regex and merges the results.
    parameters:
    ----------------
    regex_json_files_path_list: list of str
    model_results: list of dict
    text: str
    white_label_list: list of str
    return:
    ----------------
    merged_results: list of dict
    """
    merged = RegexNerParser(regex_json_files_path_list[0], text, white_label_list)
    for i in range(1, len(regex_json_files_path_list)):
        merged = RegexModelNerMerger(merged, RegexNerParser(regex_json_files_path_list[i], text, white_label_list))
    merged_results = RegexModelNerMerger(merged, model_results)
    return merged_results
