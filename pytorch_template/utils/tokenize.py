import spacy

spacy = spacy.load("en_core_web_sm")


def spacy_tokenize(x):
    return [
        tok.text
        for tok in spacy.tokenizer(x)
        if not tok.is_punct | tok.is_space
    ]
