# Author: AIMPED
# Date: 2023-March-10
# Description: Text tokenizer


def sentence_tokenizer(text:str, language:str)->list:
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

def word_tokenizer(sentences:list)->list:
    """
    Tokenize a list of sentences into words.
    parameters:
    sentences: list of str
    return:
    tokens: list of list of str
    """
    import nltk
    def split_punc(liste:list)->list:
        liste2 = []
        for i in liste:
            if  i.isalnum():
                liste2 += [i]
            else:
                liste2 += list(i)
        return liste2                           
    tokens= [split_punc(nltk.tokenize.wordpunct_tokenize(i)) for i in sentences]
    return tokens


# TODO: Test these functions
if __name__ == "__main__":
    text = "This is a test. This is only a test."
    print(sentence_tokenizer(text, "english"))
    print(word_tokenizer(sentence_tokenizer(text, "english")))