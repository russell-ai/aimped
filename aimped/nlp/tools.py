# Author: AIMPED    
# Date: 2023-March-10
# Description: Text cleaner
from nltk import sent_tokenize

def TextCleaner(text:str,
                language:str='english',
                remove_stopwords:bool=True,
                remove_short_words:int=1,
                remove_numbers:bool=True,
                remove_punctuation:bool=True,
                remove_non_ascii:bool=True,
                lower_case:bool=True)->str:
    
    """Clean text by removing unnecessary characters and altering the format of words.
    :param text: The text to be cleaned
    :param language: The language of the text
    :param remove_stopwords: Whether to remove stopwords or not
    :param remove_short_words: Whether to remove short words or not
    :param remove_numbers: Whether to remove numbers or not
    :param remove_punctuation: Whether to remove punctuation or not
    :param remove_non_ascii: Whether to remove non-ascii characters or not
    :param lower_case: Whether to convert all characters to lower case or not
    :return: The cleaned text
    """
    import re
    import nltk
    # lower case
    if lower_case: 
        text = text.lower()
    # remove punctuation
    if remove_punctuation:
        text = re.sub(r'[^\w\s]','',text)
    # remove numbers
    if remove_numbers:
        text = re.sub(r'\d+','',text)
    # remove stop words
    if remove_stopwords:
        try:
            stopwords = nltk.corpus.stopwords.words(language)
            text = ' '.join([word for word in text.split() if word not in stopwords])
        except LookupError:
            nltk.download('stopwords')
            stopwords = nltk.corpus.stopwords.words(language)
            text = ' '.join([word for word in text.split() if word not in stopwords])
        except:
            print('No stopwords for language: {}'.format(language))
            print("Please use nltk.download('stopwords') to download the stopwords for the language you want to use.")
    # remove short words
    if remove_short_words:
        text = ' '.join([word for word in text.split() if len(word) > remove_short_words])
    # remove non-ascii characters
    if remove_non_ascii:
        text = text.encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


