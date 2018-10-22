## I refactored my code using Matthias Orlikowski's code 

import spacy
import glob
import re
import pandas as pd
import os

nlp = spacy.blank('nl')
nlp.add_pipe(nlp.create_pipe('sentencizer'))

print('Pipeline: ', nlp.pipe_names)


def is_noise(token, remove_stopwords=True, remove_punctuation=True):
    token_flags = [token.is_stop, token.is_punct]
    do_flags = [remove_stopwords, remove_punctuation]
    flags = [token and do for token, do in zip(token_flags, do_flags)]
    return any(flags)


def normalize(token, do_lemmatize=True, do_lower=True):
    if do_lemmatize and do_lower:
        return token.lemma_.lower()
    if do_lemmatize:
        return token.lemma_
    if do_lower:
        return token.lower_


def extract_sentences(texts, do_lower=True, do_lemmatize=True, remove_stopwords=True, remove_punctuation=True):
    for doc in nlp.pipe(texts, batch_size=10000, n_threads=20):
        yield [[normalize(token, do_lower, do_lemmatize) for token in sent
                if not is_noise(token, remove_stopwords, remove_punctuation)]
               for sent in doc.sents]


def preprocess_year(path, start_y, end_y):
    for file_ in glob.glob(path + "/*.tsv"):
        filename = re.sub(path + '/', '', file_)
        filename = filename[12:]
        for year in range(start_y, end_y):
            if filename.startswith(str(year)):
                print(filename)
                df = pd.read_csv(
                    file_, sep='\t', encoding='utf-8', header=None)
                df = df.sample(frac=0.1, replace=False)
                df.columns = ['date', 'page', 'size', 'min_x', 'min_y',
                              'max_x', 'max_y', 'w', 'h', 'image_url', 'ocr_url', 'ocr']
                df = df[df['ocr'].notnull()]  # remove empty rows
                df['ocr'].apply(extract_sentences)
                out_path = "sentences_new/{0}.txt".format(year)
                with open(out_path, 'a+') as out_file:
                    articles = (article for article in df['ocr'].values)
                    for sentences in extract_sentences(articles, do_lemmatize=False):
                        print(*[' '.join(sentence) for sentence in sentences if sentence],
                              file=out_file,
                              sep=os.linesep)


if __name__ == '__main__':
	preprocess_year('../newspapers2018/vk/articles', 1947, 1995)
