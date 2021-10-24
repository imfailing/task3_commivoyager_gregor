import pickle
import sys
from flask import Flask, render_template, request, redirect, url_for
import logging

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


@app.route('/')
def index():
    logging.info('Rendering main page')
    return render_template('index.html')


@app.route('/', methods=['POST', 'GET'])
def get_recommendations_web():
    if request.method == 'POST':
        user_id = request.form['search']
        result = model.get_result(user_id)
        logging.info(f'Searching for user {user_id}')
        return render_template('recommendations.html', history=result['history'],
                               recommendations=result['recommendations'])


@app.route('/api')
def api():
    logging.info('Rendering api page')
    return render_template('api.html')


@app.route('/api', methods=['POST', 'GET'])
def get_recomendation():
    if request.method == 'POST':
        user_id = request.form['search']
        logging.info(f'Searching for user {user_id}')
        return redirect(url_for('get_recommendations_json', user_id=user_id))


@app.route('/api/v1.0/recommendation/<user_id>')
def get_recommendations_json(user_id):
    logging.info(f'Get recommendation for user {user_id}')
    result = model.get_result(user_id)
    logging.info(f'Get result: {result}')
    return result


@app.before_first_request
def load_model():
    logging.info('Loading model')
    try:
        global model
        model = pickle.load(open('models/final_model.sav', 'rb'))
    except:
        logging.error('Error while loading model')
        sys.exit(1)
    logging.info('Starting Web Api')


if __name__ == '__main__':
    app.run(debug=True)