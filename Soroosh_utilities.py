import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt


def mae(y_trues, y_preds):
    if not isinstance(y_trues, np.ndarray):
        y_trues = np.asarray(y_trues)

    if not isinstance(y_preds, np.ndarray):
        y_preds = np.asarray(y_preds)

    return np.mean(np.abs(y_trues-y_preds))


def rmse(y_trues, y_preds):
    if not isinstance(y_trues, np.ndarray):
        y_trues = np.asarray(y_trues)

    if not isinstance(y_preds, np.ndarray):
        y_preds = np.asarray(y_preds)

    return np.sqrt(np.mean(np.power(y_trues-y_preds, 2)))


def mrae(y_trues, y_preds):
    if not isinstance(y_trues, np.ndarray):
        y_trues = np.asarray(y_trues)

    if not isinstance(y_preds, np.ndarray):
        y_preds = np.asarray(y_preds)
    return np.mean(np.abs(np.divide(y_trues -y_preds, y_trues)))


def range_standardizer(x):
    """ Returns Range standardized data set.
    Input: a numpy array, representing entity-to-feature matrix.
    """

    x_rngs = np.ptp(x, axis=0)
    x_means = np.mean(x, axis=0)

    x_r = np.divide(np.subtract(x, x_means), x_rngs)  # range standardization

    return x_r


def zscore_standardizer(x):
    """ Returns Z-scored standardized data set.
    Input: a numpy array, representing entity-to-feature matrix.
    """

    x_stds = np.std(x, axis=0)
    x_means = np.mean(x, axis=0)

    x_z = np.divide(np.subtract(x, x_means), x_stds)  # z-scoring

    return x_z


def add_to_regression_comparison(df, y_preds, y_trues, name):
    df.at[name, 'MRAE'] = mrae(y_trues=y_trues, y_preds=y_preds)
    df.at[name, 'RMSE'] = rmse(y_trues=y_trues, y_preds=y_preds)
    df.at[name, 'R^2-Score'] = metrics.r2_score(y_trues, y_preds)
    # df.at[name, 'Predictions'] = np.asarray(y_preds)
    # df.at[name, 'Ground Truth'] = np.asarray(y_trues)

    return df


def add_to_data_generation_comparison(df, y_preds, y_trues, name):
    # Bcz some of the training samples are indeed small, and this yields the denominator of division in MRAE to zero.
    # Thus, I had to change it to MAE.
    df.at[name, 'MAE'] = mae(y_trues=y_trues, y_preds=y_preds)
    df.at[name, 'RMSE'] = rmse(y_trues=y_trues, y_preds=y_preds)
    # df.at[name, 'Predictions'] = np.asarray(y_preds)
    # df.at[name, 'Ground Truth'] = np.asarray(y_trues)
    df.to_csv("data-generation_results.csv")
    return df


def plot_loss(history, name):
    _ = plt.figure(figsize=(13.5, 7.5))
    plt.plot(history.history['loss'], label='Train Loss-' + name)
    plt.plot(history.history['val_loss'], label='Valid. Loss-' + name)
    plt.ylabel("Error")
    plt.xlabel("Epochs")
    plt.legend()
    plt.title("Train-Validation Errors for " + name)
    plt.savefig(name+".png")
    plt.show()


def plot_predictions(y_test, y_pred, name):
    _ = plt.figure(figsize=(13.5, 7.5))
    plt.scatter(y_test, y_pred)
    plt.xlabel("True Values (" + name + ")")
    plt.ylabel("True Values (" + name + ")")
    plt.title("Scatter plot of target values vs predicted values")


def plot_error_distribution(y_test, y_pred, name, n_bins):
    error = y_pred - y_test
    plt.hist(error, bins=n_bins)
    plt.xlabel("Prediction Error (" + name + ")")
    plt.ylabel('Count')


def display_samples(x, y=None):
    if not isinstance(x, (np.ndarray, np.generic)):
        x = np.array(x)
    n = x.shape[0]
    fig, axs = plt.subplots(1, n, figsize=(n, 1))
    if y is not None:
        fig.suptitle(np.argmax(y, axis=1))
    for i in range(n):
        axs.flat[i].plot(x[i].squeeze(), )
        axs.flat[i].axis('off')
    plt.show()
    plt.close()  