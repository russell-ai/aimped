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

def PreprocessText(text):
    """
    Preprocesses the given text by splitting it into paragraphs, and tokenizing each paragraph into sentences.
    If a paragraph is empty, it is replaced by ['<p>--</p>'].

    Parameters:
    text (str): The input text to be preprocessed.

    Returns:
    list: A list of strings, where each string is a preprocessed sentence with the suffix "cumle_sonu".
    """
    from nltk import sent_tokenize
    paragraphs = text.split("\n")
    sentenced_paragraphs = [sent_tokenize(paragraph) if paragraph else ['<p>--</p>'] for paragraph in paragraphs]
    return [" ".join([sentence + " cumle_sonu" for sentence in sentences]) for sentences in sentenced_paragraphs]

def split_text(text: str, max_length: int) -> list:
    """
    Splits the given text into chunks of a maximum length, making sure that sentences are not split
    across different chunks.

    Args:
        text (str): The text to be split.
        max_length (int): The maximum length of each chunk, in terms of the number of words.

    Returns:
        List[List[str]]: A list of chunks, where each chunk is represented by a list of strings (words).
    """
    from aimped.nlp.tools import PreprocessText

    input_parapgraphs = PreprocessText(text)
    input_chunks = []
    for input_paragraph in input_parapgraphs:
        temp = []
        text_tokens = input_paragraph.split()
        start_idx = 0

        while start_idx < len(text_tokens):
            end_idx = start_idx + max_length
            if end_idx > len(text_tokens):
                end_idx = len(text_tokens)
            chunk = " ".join(text_tokens[start_idx:end_idx])
            if chunk.endswith("cumle_sonu"):
                temp.append(chunk.replace(" cumle_sonu", "").replace("cumle_sonu", "").strip())
                start_idx = end_idx
            else:
                for i in range(end_idx, len(text_tokens)):
                    chunk += " " + text_tokens[i]
                    if text_tokens[i].endswith("cumle_sonu"):
                        temp.append(chunk.replace("cumle_sonu", "").strip())
                        start_idx = i + 1
                        break
        input_chunks.append(temp)
    return input_chunks

