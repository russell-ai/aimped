# **aimped**
![aimped](https://dev.aimped.ai/static/media/AimpedBirdLogo.0b3c7cc26d31afe7bd73db496acb01d1.svg)

**Aimped is a unique python library that provides classes and functions for only exclusively business-tailored AI-based NLP models.**  

# Installation  
```python
pip install aimped
```

# Usage  
```python  
import aimped
print(aimped.__version__)
```
## Examples  

### Example 1

```python  
from aimped import nlp

result = nlp.sentence_tokenizer("Hi, welcome to aimped. Explore ai models.",language="english")
print(result)
# ['Hi, welcome to aimped.', 'Explore ai models.']
```

### Example 2
```python  
from aimped.nlp.tokenizer import sentence_tokenizer

result = sentence_tokenizer("Hi, welcome to aimped. Explore ai models.",language="english")
print(result)
# ['Hi, welcome to aimped.', 'Explore ai models.']
```

### Example 3
```python  
from aimped.nlp.pipeline import Pipeline

pipe = Pipeline(model=model, tokenizer=tokenizer, device='cpu')
result = pipe.ner_result(
                        text=text,
                        sents_tokens_list=sents_tokens_list,
                        sentences=sentences)
print(result)
```

