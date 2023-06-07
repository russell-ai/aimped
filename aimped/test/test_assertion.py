from transformers import (AutoModelForSequenceClassification,
                          AutoModelForTokenClassification, AutoTokenizer,
                          pipeline)
from aimped.nlp.pipeline import Pipeline
from aimped.nlp.tokenizer import sentence_tokenizer, word_tokenizer

text = '''Patient has a headache for the last 2 weeks, needs to get a head CT, and appears anxious when she walks fast. 
No alopecia noted. She denies pain.
Patient has a headache for the last 2 weeks, needs to get a head CT, and appears anxious when she walks fast. 
No alopecia noted. She denies pain.'''

as_model_path = r"C:\Users\rcali\Desktop\kubeflow\ner_assertion\assertion_model"
as_tokenizer = AutoTokenizer.from_pretrained(as_model_path)
as_model = AutoModelForSequenceClassification.from_pretrained(as_model_path)
classifier = pipeline(task="sentiment-analysis", model=as_model, tokenizer=as_tokenizer, device=-1)

ner_model_path = r"C:\Users\rcali\Desktop\kubeflow\ner_assertion\ner_model"
ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_path)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_path)

ner_pipe = Pipeline(model=ner_model, tokenizer=ner_tokenizer, device='cpu')

sentences = sentence_tokenizer(text, "english")
sents_tokens_list = word_tokenizer(sentences)
white_label_list = ['problem', 'test', 'treatment']
assertion_white_label_list = ['present', 'absent', 'possible']

# print("sentences: ", sentences)
# print("sents_tokens_list: ", sents_tokens_list)

tokens, preds, probs, begins, ends, sent_begins, sent_ends, sent_idxs = ner_pipe.ner_result(
    text=text,
    sents_tokens_list=sents_tokens_list,
    sentences=sentences,
    assertion_relation=True
)
# print(tokens, preds, probs, begins, ends, sent_begins, sent_ends, sent_idxs)

ner_results = ner_pipe.chunker_result(text=text,
                                      white_label_list=white_label_list,
                                      tokens=tokens,
                                      preds=preds,
                                      probs=probs,
                                      begins=begins,
                                      ends=ends,
                                      assertion_relation=True,
                                      sent_begins=sent_begins,
                                      sent_ends=sent_ends,
                                      sent_idxs=sent_idxs)

# print(ner_results)

results = ner_pipe.assertion_result(ner_results=ner_results,
                                    classifier=classifier,
                                    assertion_white_label_list=assertion_white_label_list,
                                    sentences=sentences,
                                    )

print(results)
