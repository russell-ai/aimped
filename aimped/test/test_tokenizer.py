from aimped.nlp.tokenizer import SentenceTokenizer, WordTokenizer


def test_sentence_tokenizer():
    text = "This is a test. This is a test."
    sents = SentenceTokenizer(text, language="english")
    print(sents)
    assert sents == ['This is a test.', 'This is a test.']

def test_word_tokenizer():
    text = "This is a test. This is a test."
    words = WordTokenizer(SentenceTokenizer(text, language="english"))
    print(words)
    assert words == [['This', 'is', 'a', 'test', '.'], ['This', 'is', 'a', 'test', '.']]

if __name__ == "__main__":
    test_sentence_tokenizer()
    test_word_tokenizer()
