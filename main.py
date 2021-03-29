import pickle

import flask_monitoringdashboard as dashboard
import numpy as np
from flask import Flask, request, render_template
from flask import Response
from flask_cors import CORS, cross_origin

from apps.core.config import Config
from apps.prediction.predict_model import PredictModel
from apps.training.train_model import TrainModel

app = Flask(__name__)
dashboard.bind(app)
CORS(app)


@app.route('/training', methods=['POST'])
@cross_origin()
def training_route_client():
    """
    * method: training_route_client
    * description: method to call training route
    * return: none
    *
    * Parameters
    *   None
    """
    try:
        config = Config()
        # get run id
        run_id = config.get_run_id()
        data_path = config.training_data_path
        # trainmodel object initialization
        trainModel = TrainModel(run_id, data_path)
        # training the model
        trainModel.training_model()
        return Response("Training successfull! and its RunID is : " + str(run_id))
    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)


@app.route('/batchprediction', methods=['POST'])
@cross_origin()
def batch_prediction_route_client():
    """
    * method: batch_prediction_route_client
    * description: method to call batch prediction route
    * return: none
    *
    *
    * Parameters
    *   None
    """
    try:
        config = Config()
        # get run id
        run_id = config.get_run_id()
        data_path = config.prediction_data_path
        # prediction object initialization
        predictModel = PredictModel(run_id, data_path)
        # prediction the model
        predictModel.batch_predict_from_model()
        return Response("Prediction successfull! and its RunID is : " + str(run_id))
    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)


@app.route('/', methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])  # route to show the predictions in a web UI
@cross_origin()
def index():
    """
        * method: single_prediction
        * description: method to call batch prediction route
        * return: none
        *
        *
        * Parameters
        *   None
        """
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            prior_default = int(request.form['prior_default'])
            years_employed = int(request.form['years_employed'])
            credit_score = int(request.form['credit_score'])
            income = float(request.form['income'])

            filename = 'RandomForest.sav'
            loaded_model = pickle.load(open(filename, 'rb'))  # loading the model file from the storage
            data = np.array([[prior_default, years_employed, credit_score, income]])
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
    app.run()
    # app.run(debug=True)
    # host = '0.0.0.0'
    # port = 5000
    # httpd = simple_server.make_server(host, port, app)
    # httpd.serve_forever()

