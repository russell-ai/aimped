import torch
from aimped.nlp.tokenizer import SentenceTokenizer, WordTokenizer
from transformers import AutoTokenizer, AutoModelForTokenClassification
from aimped.nlp.pipeline import Pipeline

# load model
saved_model_path = r"C:\Users\rcali\Desktop\kubeflow\deid-ner-gitlab\model"
tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
model = AutoModelForTokenClassification.from_pretrained(saved_model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"device: {device}")

# load data
text = open("data.txt", "r").read()
sentences = SentenceTokenizer(text, "english")
sents_tokens_list = WordTokenizer(sentences)
white_label_list = ['PATIENT', 'ORGANIZATION', 'SSN', 'SEX', 'DOCTOR', 'HOSPITAL', 'AGE', 'MEDICALRECORD', 'ZIP',
                    'STREET', 'EMAIL', 'DATE', 'ID', 'CITY', 'COUNTRY', 'PROFESSION']
# print(sents_tokens_list)
pipe = Pipeline(model=model, tokenizer=tokenizer, device='cpu')
tokens, preds, probs, begins, ends = pipe.ner_result(text=text,
                                                     sents_tokens_list=sents_tokens_list,
                                                     sentences=sentences)

merged_chunks = pipe.chunker_result(text, white_label_list, tokens, preds, probs, begins, ends)
print("merged_chunks: ", merged_chunks)

