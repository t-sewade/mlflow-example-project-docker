import warnings
import sys
import os
from os.path import join

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn


def eval_metrics(actual, pred):
    """ Evaluation metrics for the model. """
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    # arguments
    data_dir = sys.argv[1]
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    l1_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = pd.read_csv(join(data_dir, 'train_x.csv'))
    test_x = pd.read_csv(join(data_dir, 'test_x.csv'))
    train_y = pd.read_csv(join(data_dir, 'train_y.csv'))
    test_y = pd.read_csv(join(data_dir, 'test_y.csv'))

    with mlflow.start_run():
        print('Training Model...')
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        print('Evaluating Model...')
        predicted_qualities = lr.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        # log results
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(lr, "model")
