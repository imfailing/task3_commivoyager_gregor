import random
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class BooksModel:
    def __init__(self, books, users):
        self.books = books
        self.users = users

        self.cosine_sim = self.__get_cosine_sim()
        self.i_cold_start, self.j_cold_start = np.where(self.cosine_sim == self.__find_min_in_arr(self.cosine_sim))
        self.help_i_cold_start = pd.Series(self.books['id'], index=self.books.index).drop_duplicates()
        self.top_books_collapse_fields = (self.users['smart_collapse_field'].value_counts()[:10]).index.tolist()
        self.i_with_user_hist = pd.Series(self.books.index, index=self.books['id']).drop_duplicates()

    def __find_min_in_arr(self, arr):
        return min(map(min, arr))

    def __get_cosine_sim(self):
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.books['corpus'])
        return linear_kernel(tfidf_matrix, tfidf_matrix)

    def user_history(self, user_id):
        user_hist = self.users[self.users['user_id'] == user_id]
        if len(user_hist) > 0:
            return user_hist.sort_values(by=['dt'])['book_id'].values.tolist()
        else:
            return []

    def get_recommendations_for_book(self, book_id, number=10):
        idx = self.i_with_user_hist[book_id]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1: number + 1]  # c 1 потому что 0 это эта книга
        books_indices = [i[0] for i in sim_scores]
        return self.books['id'].iloc[books_indices].tolist()

    def user_without_history_recommendations(self):
        recommendations = []
        top_books_ids = self.users[self.users['smart_collapse_field'].isin(self.top_books_collapse_fields)] \
            [['book_id', 'smart_collapse_field']] \
            .drop_duplicates(subset='smart_collapse_field', keep="last")['book_id'] \
            .to_list()

        random_vals1 = random.randint(0, len(top_books_ids) - 1)
        recommendations.append(
            top_books_ids[random_vals1])
        top_books_ids.pop(random_vals1)
        random_vals2 = random.randint(0, len(top_books_ids) - 1)
        recommendations.append(
            top_books_ids[random_vals2])

        random_vals3 = random.randint(0, len(self.i_cold_start) - 1)
        recommendations.append(
            self.help_i_cold_start[self.i_cold_start[random_vals3]])
        recommendations.append(
            self.help_i_cold_start[self.j_cold_start[random_vals3]])
        val4 = (self.cosine_sim[self.i_cold_start[random_vals3]] + self.cosine_sim[self.j_cold_start[random_vals3]])
        recommendations.append(
            self.help_i_cold_start[
                np.where(val4 == val4.min())[0][random.randint(0, len(np.where(val4 == val4.min())[0]) - 1)]])

        return recommendations

    def user_with_history_recommendations(self, user_id):
        """ Если user_hist < 0, то по первой книге будет рекомендоваться больше книг.
            Пример:
                user_hist = 3
                рекомендация по 1-ой книге - 3
                рекомендация по 2-ой книге - 1
                рекомендация по 3-ой книге - 1
        """
        recommendations = []
        user_hist = self.users[self.users['user_id'] == user_id] \
            .groupby(by="smart_collapse_field").first() \
            .sort_values("dt", ascending=False)
        user_hist_book_ids = user_hist["book_id"].to_list()

        last_5_len = len(user_hist_book_ids[:5])
        n_rec = np.ones(last_5_len).astype(int)
        n_rec[0] = n_rec[0] + 5 - last_5_len

        for n, book_id in zip(n_rec, user_hist_book_ids):
            def recursive_recommendations(number_of_recs, coeff):
                recs = self.get_recommendations_for_book(book_id, number=10 * coeff)
                recs_unique = np.setdiff1d(recs, user_hist_book_ids)
                recs_unique = np.setdiff1d(recs_unique, recommendations)
                if len(recs_unique) < number_of_recs:
                    return recursive_recommendations(number_of_recs, coeff + 1)
                return recs_unique[:number_of_recs]
            recommendations = recommendations + recursive_recommendations(n, 1).tolist()

        return recommendations

    def get_recommendations_for_user(self, user_id):
        user_history = self.user_history(user_id)
        if len(user_history) > 0:
            recommendations = self.user_with_history_recommendations(user_id)
            return recommendations, user_history[:5]
        else:
            recommendations = self.user_without_history_recommendations()
            return recommendations, user_history[:5]

    def get_book_for_id(self, id):
        id = int(id)
        return {"id": id,"title": self.books.loc[self.books['id'] == id, 'title_orig'].item(),"author": self.books.loc[self.books['id'] == id, 'author_fullName'].item()}

    def get_result(self, user_id):
        recommendations, user_history = self.get_recommendations_for_user(int(user_id))
        return {"recommendations": [self.get_book_for_id(rec_book) for rec_book in recommendations],
              "history": [self.get_book_for_id(hist_book) for hist_book in user_history]}