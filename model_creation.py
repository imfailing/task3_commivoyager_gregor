import logging
import pickle
import pandas as pd
import argparse
from services.data_preparation import BooksModelData
from services.model_integration import BooksModel

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--books", default='data/books.jsn', type=str)
parser.add_argument("--users", default='data/dataset_knigi_1.xlsx', type=str)
parser.add_argument("--model_dataset", default='data/model_books.csv', type=str)
parser.add_argument("--model", default='models/final_model.sav', type=str)
args = parser.parse_args()

if __name__ == '__main__':
    logging.info('Starting creating model')
    try:
        initial_books = pd.read_json(args.books)
        initial_users = pd.read_excel(args.users, sheet_name="Лист1", engine='openpyxl')
        path_to_books = args.model_dataset
        data = BooksModelData()
        books = data.make_and_save_books_dataset(initial_books, path_to_books)
        users = data.load_users_dataset(initial_users, initial_books)
        model = BooksModel(books, users)
        pickle.dump(model, open(args.model, 'wb'))
    except:
        logging.error('Error creating model')
    finally:
        logging.info('Model updated successfully')
