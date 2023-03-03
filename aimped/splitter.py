
def sentence_splitter(text, delimiter='.'):
    """Split text into sentences."""
    sentences = text.split(delimiter)
    return sentences


if __name__ == "__main__":
    text = "This is a test. This is only a test."
    print(sentence_splitter(text))