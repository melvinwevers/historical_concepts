import pandas as pd


'''
The following resources have been used:
    - Dutch Wordnet: https://github.com/cltl/OpenDutchWordnet
    - Opentaal
    - Target word lists
    - Questions words from Dutch Embeddings: Question words
    taken from https://github.com/clips/dutchembeddings
Use Dutch wordnet as well as words from target word lists
'''

with open('../dictionary.dict') as f:
        dictionary_nl = f.readlines()
        dictionary_nl = [x.strip() for x in dictionary_nl]

with open('../dutchembeddings/data/question-words.txt') as f:
    '''
    Question words taken from https://github.com/clips/dutchembeddings
    '''
    question_words = f.readlines()

words_questions = []

for row in question_words:
    if row.startswith(":"):
        pass
    else:
        words = row.split(" ")
        for word in words:
            if word not in words_questions:
                word = word.strip()
                words_questions.append(word)

open_taal_words = []
with open('../OpenTaal-210G-woordenlijsten/OpenTaal-210G-verwarrend.txt') as f:
    '''
    Use opentaal dictionaries
    '''
    open_taal = f.read().splitlines()

open_taal_words = open_taal_words + open_taal

dictionary_nl.remove('')

for word in words_questions:
    if word not in dictionary_nl:
        dictionary_nl.append(word)

for word in open_taal_words:
    if word not in dictionary_nl:
        dictionary_nl.append(word)

with open('nl_dict2.dict', 'w') as f:
    for item in dictionary_nl:
        f.write("%s\n" % item)