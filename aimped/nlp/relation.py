# Author AIMPED
# Date 2023-March-14
# Description This file contains the pipeline for relation extraction.

def relation_annotate_sentence(df):
    """It annotates the sentence with [Entity] tags.
    parameters:
    ----------------
    df: pandas dataframe
    return:
    ----------------
    df['new_sentence']: str
    """

    a1_start = min(df['sent_begin1'] - 1, df['sent_begin2'])
    a1_end = min(df['sent_end1'] + 1, df['sent_end2'] + 1)

    a2_start = max(df['sent_begin1'] - 1, df['sent_begin2'])
    a2_end = max(df['sent_end1'] + 1, df['sent_end2'] + 1)

    df['new_sentence'] = " ".join([
        df['sentence'][:a1_start],
        'e1b',
        df['sentence'][a1_start:a1_end],
        'e1e',
        df['sentence'][a1_end:a2_start],
        'e2b',
        df['sentence'][a2_start:a2_end],
        'e2e',
        df['sentence'][a2_end:]
    ])

    return df['new_sentence']


def RelationResults(sentences, ner_chunk_results, relation_classifier,
                    ner_white_label_list, relation_white_label_list, one_to_many,
                    one_label, return_svg):
    """It returns the relation results of a text.
    parameters:
    ----------------
    sentences: list of str
    ner_chunk_results: list of dict
    relation_classifier: str
    ner_white_label_list: list of str
    relation_white_label_list: list of str
    one_to_many: bool
    one_label: str
    return_svg: bool
    return:
    ----------------
    results: list of dict
    """
    import itertools
    import pandas as pd
    import numpy as np
    end_df = pd.DataFrame()
    df_ner_chunk_results = pd.DataFrame(ner_chunk_results)
    for i in df_ner_chunk_results.sent_idx.unique():
        sentence_bazli_ner_results = df_ner_chunk_results[df_ner_chunk_results.sent_idx == i]
        sentence_bazli_ner_results = sentence_bazli_ner_results.values.tolist()
        if len(sentence_bazli_ner_results) >= 2:
            df = pd.DataFrame(itertools.combinations(sentence_bazli_ner_results, 2))
            df['firstCharEnt1'] = df[0].apply(lambda x: x[2])
            df['lastCharEnt1'] = df[0].apply(lambda x: x[3])
            df['entity1'] = df[0].apply(lambda x: x[0])
            df['chunk1'] = df[0].apply(lambda x: x[1])
            df['sent_begin1'] = df[0].apply(lambda x: x[-2])
            df['sent_end1'] = df[0].apply(lambda x: x[-1])
            df['firstCharEnt2'] = df[1].apply(lambda x: x[2])
            df['lastCharEnt2'] = df[1].apply(lambda x: x[3])
            df['entity2'] = df[1].apply(lambda x: x[0])
            df['chunk2'] = df[1].apply(lambda x: x[1])
            df['sent_begin2'] = df[1].apply(lambda x: x[-2])
            df['sent_end2'] = df[1].apply(lambda x: x[-1])

            if one_to_many:
                df = df[((df.entity1 == one_label) | (df.entity2 == one_label))]
                df = df[~((df.entity1 == one_label) & (df.entity2 == one_label))]
            else:
                for ner_label in ner_white_label_list:
                    df = df[~((df.entity1 == ner_label) & (df.entity2 == ner_label))]

            if len(df) != 0:
                df['sentID'] = i
                df['sentence'] = sentences[i]
                df = df.drop([0, 1], axis=1)
                df['new_sentence'] = np.nan
                df['new_sentence'] = df.apply(relation_annotate_sentence, axis=1)
                df.reset_index(drop=True, inplace=True)
                rel_results = relation_classifier(list(df['new_sentence']))
                df = pd.concat([df, pd.DataFrame(rel_results)], axis=1)
                df = df[['sentID', 'sentence', 'firstCharEnt1', 'sent_begin1', 'lastCharEnt1', 'sent_end1', 'entity1',
                         'chunk1',
                         'firstCharEnt2', 'sent_begin2', 'lastCharEnt2', 'sent_end2', 'entity2', 'chunk2', 'label',
                         'score'
                         ]]
                df = df[df['label'].isin(relation_white_label_list)]
                end_df = pd.concat([end_df, df], ignore_index=True)

    if end_df.empty or return_svg:
        return end_df.to_dict(orient='records')

    else:
        return end_df[['firstCharEnt1', 'lastCharEnt1', 'entity1', 'chunk1',
                       'firstCharEnt2', 'lastCharEnt2', 'entity2', 'chunk2', 'label', 'score']].to_dict(
            orient='records')
