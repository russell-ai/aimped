# Author: AIMPED
# Date: 2023-March-10
# Description: Text tokenizer


def sentence_tokenizer(text: str, language: str) -> list:
    """ 
    Tokenize a text into sentences.
    text: str    
    language: str (turkish, swedish, spanish, slovene, 
            portuguese, polish, norwegian, italian, 
            greek, german, french, finnish, estonian 
            english, dutch, danish, czech )
    """
    import nltk
    sent = nltk.tokenize.sent_tokenize(text, language=language)
    return sent


def word_tokenizer(sentences: list) -> list:
    """
    Tokenize a list of sentences into words.
    parameters:
    sentences: list of str
    return:
    tokens: list of list of str
    """
    import nltk
    tokens = [nltk.tokenize.wordpunct_tokenize(i) for i in sentences]
    return tokens

