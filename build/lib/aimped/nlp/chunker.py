# Author: AIMPED
# Date: 2023-March-11
# Description: Chunk merger


def ChunkMerger(text, white_label_list, tokens, preds, probs, begins, ends,
                assertion_relation=False, sent_begins=[], sent_ends=[], sent_idxs=[]):
    """
    Parameters
    ----------
    text : str
    white_label_list : list
    tokens : list
    preds : list
    probs : list
    begins : list
    ends : list
    assertion_relation : bool, optional
        The default is False.
    sent_begins : list, optional
        The default is []. Only used if assertion_relation is True
    sent_ends : list, optional
        The default is []. Only used if assertion_relation is True
    sent_idxs : list, optional
        The default is []. Only used if assertion_relation is True

    Returns
    -------
    results : list
    """
    import numpy as np
    results = []
    idx = 0
    while idx < len(tokens):
        label = preds[idx]
        if label != "O": label = label[2:]
        if label in white_label_list:
            all_scores = []
            if preds[idx][2:] == label and preds[idx - 1 if idx != 0 else idx][2:] != label or idx == 0:
                all_scores.append(probs[idx])
                if assertion_relation:
                    begin, end, sent_begin, sent_end = begins[idx], ends[idx], sent_begins[idx], sent_ends[idx]
                else:
                    begin, end = begins[idx], ends[idx]
            if preds[idx][2:] == label and preds[idx - 1 if idx != 0 else idx][2:] == label and idx != 0:
                while idx < len(tokens) and preds[idx][2:] == label:
                    all_scores.append(probs[idx])
                    if assertion_relation:
                        end, sent_end = ends[idx], sent_ends[idx]
                    else:
                        end = ends[idx]
                    idx += 1
                idx -= 1

            if idx == len(tokens) - 1:
                if not assertion_relation:
                    confidence = np.mean(all_scores).item()
                    chunk = text[begin:end]
                    results.append({"entity": label,
                                    "confidence": confidence,
                                    "chunk": chunk,
                                    "begin": int(begin),
                                    "end": int(end)})
                else:
                    chunk = text[begin:end]
                    results.append({"entity": label,
                                    "chunk": chunk,
                                    "begin": int(begin),
                                    "end": int(end),
                                    "sent_idx": int(sent_idxs[idx]),
                                    "sent_begin": int(sent_begin),
                                    "sent_end": int(sent_end)})


            elif idx < len(tokens) - 1:
                if preds[idx + 1][2:] != label:
                    if not assertion_relation:
                        confidence = np.mean(all_scores).item()
                        chunk = text[begin:end]
                        results.append({"entity": label,
                                        "confidence": confidence,
                                        "chunk": chunk,
                                        "begin": int(begin),
                                        "end": int(end)})
                    else:
                        chunk = text[begin:end]
                        results.append({"entity": label,
                                        "chunk": chunk,
                                        "begin": int(begin),
                                        "end": int(end),
                                        "sent_idx": int(sent_idxs[idx]),
                                        "sent_begin": int(sent_begin),
                                        "sent_end": int(sent_end)})

        idx += 1
    return results


