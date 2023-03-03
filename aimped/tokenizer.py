from .splitter import sentence_splitter


def tokenizer(text):
    return [token for sentence in sentence_splitter(text) for token in sentence.split()]



if __name__ == "__main__":
    text = "This is a test. This is only a test."
    print(tokenizer(text))