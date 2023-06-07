from transformers import (AutoModelForSequenceClassification,
                          AutoModelForTokenClassification, AutoTokenizer,
                          pipeline)
from aimped.nlp.pipeline import Pipeline
from aimped.nlp.tokenizer import sentence_tokenizer, word_tokenizer

# %%
ner_model_path = r"C:\Users\rcali\Desktop\kubeflow\relation\ner_model"
ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_path)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_path)

# %%
cls_model_path = r"C:\Users\rcali\Desktop\kubeflow\relation\cls_model"
cls_tokenizer = AutoTokenizer.from_pretrained(cls_model_path)
cs_model = AutoModelForSequenceClassification.from_pretrained(cls_model_path)

classifier = pipeline(task="sentiment-analysis", model=cs_model, tokenizer=cls_tokenizer, device=-1)
# %%
rel_model_path = r"C:\Users\rcali\Desktop\kubeflow\relation\rel_model"
rel_tokenizer = AutoTokenizer.from_pretrained(rel_model_path)
rel_model = AutoModelForSequenceClassification.from_pretrained(rel_model_path)

relation_classifier = pipeline(task="sentiment-analysis", model=rel_model, tokenizer=rel_tokenizer, device=-1)

# %%

text = '''Successful treatment with carbimazole	of a hyperthyroid pregnancy with  hepatic impairment  after propylthiouracil administration.
A 37-year-old female patient was diagnosed with hepatic impairment 8 months ago and medical treatment with carbimazole, low-dose corticosteroids and 5-ASA was started.
Successful treatment with carbimazole	of a hyperthyroid pregnancy with  hepatic impairment  after propylthiouracil administration.'''

pipe = Pipeline(model=ner_model, tokenizer=ner_tokenizer, device='cpu')
sentences = sentence_tokenizer(text, "english")
print(sentences)
sentences = [sentences[idx] for idx, label in enumerate(classifier(sentences)) if label['label'] == 'Positive']
sents_tokens_list = word_tokenizer(sentences)
print(sents_tokens_list)
white_label_list = ['DRUG', 'ADE']

tokens, preds, probs, begins, ends, sent_begins, sent_ends, sent_idxs = pipe.ner_result(text=text,
                                                                                        sents_tokens_list=sents_tokens_list,
                                                                                        sentences=sentences,
                                                                                        assertion_relation=True)
print("tokens: ", tokens)
print("preds: ", preds)
print("probs:", probs)
print("begins:", begins)
print("ends:", ends)
# %%
results = pipe.chunker_result(text=text,
                              white_label_list=white_label_list, tokens=tokens, preds=preds, probs=probs, begins=begins,
                              ends=ends,
                              sent_begins=sent_begins, sent_ends=sent_ends, sent_idxs=sent_idxs, assertion_relation=True)
print("results: ", results)

# %%
relation_white_label_list = ['Positive', 'Negative']
results = pipe.relation_result(sentences=sentences, ner_chunk_results=results, relation_classifier=relation_classifier,
                               ner_white_label_list=white_label_list,
                               relation_white_label_list=relation_white_label_list,
                               one_to_many=True, one_label='DRUG', return_svg=False)
print("results: ", results)
