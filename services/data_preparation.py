import pandas as pd
import nltk
from nltk import sent_tokenize, regexp_tokenize
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer


class BooksModelData:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')

    def __get_id_from_url(self, string):
        return int(string.split('/')[5])

    def __normalize_tokens(self, tokens):
        morph = MorphAnalyzer()
        return [morph.parse(tok)[0].normal_form for tok in tokens]

    def __remove_stopwords(self, tokens, stopwords=None, min_length=4):
        if not stopwords:
            return tokens
        stopwords = set(stopwords)
        tokens = [tok
                  for tok in tokens
                  if tok not in stopwords and len(tok) >= min_length]
        return tokens

    def __tokenize_n_lemmatize(self, text, stopwords=None, normalize=True, regexp=r'(?u)\b\w{4,}\b'):
        words = [w for sent in sent_tokenize(text)
                 for w in regexp_tokenize(sent, regexp)]
        if normalize:
            words = self.__normalize_tokens(words)
        if stopwords:
            words = self.__remove_stopwords(words, stopwords)
        return words

    def make_books_dataset(self, initial_books):
        books_dataset_features = [
            'id',
            'title_orig',
            'author_fullName',
            'rubric_name',
            'publisher_name',
            'serial_name',
            'annotation'
        ]

        books = initial_books[books_dataset_features]
        books = books.fillna('')
        books['all'] = \
            books['title_orig'] + ' ' + \
            books['author_fullName'] + ' ' + \
            books['rubric_name'] + ' ' + \
            books['publisher_name'] + ' ' + \
            books['serial_name'] + ' ' + \
            books['annotation']
        books['corpus'] = books['all'] \
            .apply(lambda x: ' '.join(self.__tokenize_n_lemmatize(x, stopwords=stopwords.words('russian'))))

        return books

    def save_books_dataset(self, df, save_path):
        df.to_csv(save_path, index=False, sep='\t', encoding='utf8')

    def load_books_dataset(self, read_path):
        return pd.read_csv(read_path, sep='\t', encoding='utf8')

    def make_and_save_books_dataset(self, initial_books, save_path):
        df = self.make_books_dataset(initial_books)
        self.save_books_dataset(df, save_path)
        return df

    def add_and_save_new_data_to_books_dataset(self, books, initial_new_data):
        new_data_diff = initial_new_data.merge(books, on='id', how='outer', suffixes=['', '_'], indicator=True)
        new_data_diff = self.make_books_dataset(new_data_diff)
        new_books = books.append(new_data_diff, ignore_index=True)
        self.save_books_dataset(new_books)

    def load_users_dataset(self, initial_users, initial_books):
        initial_users['book_id'] = initial_users["source_url"].apply(lambda v: self.__get_id_from_url(v))
        additional_user_rows = \
            initial_books[['id', 'smart_collapse_field', 'title_orig', 'author_fullName']] \
            .rename(columns={"id": "book_id"})
        users = initial_users \
            .merge(additional_user_rows, how='inner', on='book_id') \
            .sort_values(by=['user_id', 'dt'])
        return users
