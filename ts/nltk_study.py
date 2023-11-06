from nltk.tokenize import word_tokenize
from nltk.text import Text


if __name__ == "__main__":

    str1 = "Today's weather is good. very windy and sunny, "\
            "we have no classes in the afternoon. We have to play basketball tomorrow."
    ss = word_tokenize(str1)
    print(ss)

