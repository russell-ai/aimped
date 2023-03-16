# Author: AIMPED
# Date: 2023-March-12
# Description: This file contains the pipeline for de-identification of clinical notes


def maskText(merged, text):
    """
    It masks the actual chunks in the text with their entity labels.
    parameters:
    ----------------
    merged: list of dict
    text: str
    return:
    ----------------
    masked_text: str
    """
    text_mask = text[:]
    for i in range(len(merged) - 1, -1, -1):
        text_mask = text_mask[:merged[i]['begin']] + f"<<{merged[i]['entity']}>>" + text_mask[merged[i]['end']:]
    return text_mask


def fakedChunk(fake_csv_path, merged ):
    """ 
    Randomly select entities from fake.csv file and add them to entities list align with true labels.
    parameters:
    ----------------
    fake_csv_path: str
    merged: list of dict
    return:
    ----------------
    merged: list of dict
    """
    import pandas as pd
    import random

    fake_df = pd.read_csv(fake_csv_path, sep=',', encoding='utf8')
    for item in merged:
        item["faked_chunk"] = str(random.choice(fake_df[item['entity']]))
    return merged


def fakedText(merged, text):
    """
    Rewrite the text with faked chunks that comes from fakedChunk function.
    parameters:
    ----------------
    merged: list of dict
    text: str
    return:
    ----------------
    faked_text: str
    """
    
    faked_text = text[:]
    for i in range(len(merged) - 1, -1, -1):
        faked_text = faked_text[:merged[i]['begin']] + f"{merged[i]['faked_chunk']}" + faked_text[merged[i]['end']:]
    return faked_text


def deidentification(faked, masked, merged_results, text, fake_csv_path):
    """
    It masks the actual chunks in the text with their entity labels.
    parameters:
    ----------------
    faked: bool
    masked: bool
    merged_results: list of dict
    text: str
    fake_csv_path: str
    return:
    ----------------
    entities: list of dict
    masked_text: str
    faked_text: str
    """
    if faked and masked:
        text_mask = maskText(merged=merged_results, text=text)
        entities_with_faked_chunks = fakedChunk(fake_csv_path=fake_csv_path, merged=merged_results)
        faked_text = fakedText(merged=entities_with_faked_chunks, text=text)
        return {"entities": entities_with_faked_chunks, "masked_text":text_mask,"faked_text":faked_text}
    elif faked:
        entities_with_faked_chunks = fakedChunk(fake_csv_path=fake_csv_path,merged=merged_results)
        faked_text = fakedText(merged=entities_with_faked_chunks, text=text)
        return {"entities": entities_with_faked_chunks, "faked_text":faked_text}
    elif masked:
        text_mask = maskText(merged=merged_results, text=text)
        return {"entities": merged_results, "masked_text":text_mask}

    else:
        return {"entities": merged_results}
    

