# Author AIMPED
# Date 2023-March-12
# Description This file contains the pipeline for assertion detection of clinical notes


def assertion_annotate_sentence(df):
    """
    It annotates the sentence with [Entity] tags.   
    parameters:
    ----------------
    df: pandas dataframe
    return:
    ----------------
    df['new_sentence']: str
    """    
    import pandas as pd

    df['new_sentence'] = " ".join([
        df['sentence'][:df['sent_begin']], 
        ' [Entity] ', 
        df['sentence'][df['sent_begin']:df['sent_end']],
        ' [Entity] ', 
        df['sentence'][df['sent_end']:]])
    return df['new_sentence'] 


def AssertionModelResults(ner_results, sentences, classifier, assertion_white_label_list):
    """
    It returns the assertion detection results of a text.
    parameters:
    ----------------
    text: str
    ner_results: list of dict
    sentences: list of str
    tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizer
    model: transformers.modeling_utils.PreTrainedModel
    classifier: transformers.modeling_utils.PreTrainedModel
    return:
    ----------------
    df.to_dict(orient = 'records'): list of dict
    """
    import pandas as pd
    import numpy as np
    df = pd.DataFrame()
    if len(ner_results) >= 1:
        ner_results = list(map(lambda x : list(x.values()),ner_results))
        df = pd.DataFrame(ner_results)
        df.columns = ['ner_label','chunk', 'begin', 'end','sent_idx', 'sent_begin', 'sent_end' ]
        if len(df) != 0:
            df['sentence'] = np.nan
            for i in df.sent_idx.unique():
                df.loc[df[df.sent_idx == i].index, 'sentence'] = sentences[i]
            df['new_sentence'] = np.nan
            df['new_sentence'] = df.apply(assertion_annotate_sentence, axis = 1)
            df.reset_index(drop=True,inplace=True)
            rel_results = classifier(list(df['new_sentence']))
            df = pd.concat([df,pd.DataFrame(rel_results)], axis= 1)
            df = df[['begin', 'end', 'ner_label', 'chunk', 'label','score']]
            df.columns = ['begin','end', 'ner_label', 'chunk', 'assertion','score']
            df =df[df['assertion'].isin(assertion_white_label_list)]
    return df.to_dict(orient = 'records')



