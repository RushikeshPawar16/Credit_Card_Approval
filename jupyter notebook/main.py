## importing the necessary dependencies
from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import pickle
import numpy as np
app = Flask(__name__)  # initializing a flask app


@app.route('/', methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])  # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            prior_default_t = int(request.form['prior_default_t'])
            years_employed = int(request.form['years_employed'])
            credit_score = int(request.form['credit_score'])
            income = float(request.form['income'])

            filename = 'finalized_model_RF.sav'
            loaded_model = pickle.load(open(filename, 'rb'))  # loading the model file from the storage
            data = np.array([[prior_default_t, years_employed, credit_score, income]])
            my_prediction = loaded_model.predict(data)
            # showing the prediction results in a UI
            return render_template('result.html', prediction=my_prediction)
        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')


if __name__ == "__main__":
    # app.run(host='127.0.0.1', port=8001, debug=True)
    app.run(debug=True)  # running the app
