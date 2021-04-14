import os
import shap
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from joblib import dump, load
from Soroosh_utilities import *
import tensorflow_probability as tfp
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def apply_gini(estimator_important_features,
               pld_complete, name, thr=.001):
    important_features_indices = np.where(estimator_important_features >= thr)

    important_features_names = list(pld_complete.columns[important_features_indices])
    important_features_values = \
        estimator_important_features[np.where(estimator_important_features >= thr)].reshape(1, -1)

    indices = np.argsort(important_features_values)[0][::-1].tolist()

    print("Gini-index important Features w.r.t " + name + " Reg.!   ",)
    #     for i in indices:
    #         print("%.3f" % important_features_values[0][i], important_features_names[i], )

    return indices, important_features_values, important_features_names


def apply_permutation(estimator_important_features,
                      pld_complete, name, thr=.001):
    #     print("esti:", estimator_important_features)
    #     print("mean:", set(np.where(estimator_important_features['importances_mean'] >= thr)[0]))

    #     print("std:", set(np.where(estimator_important_features['importances_std'] < .2)[0]))

    features_indices = np.asarray(
        list(set(np.where(estimator_important_features['importances_mean'] >= thr)[0]).intersection(
            set(np.where(estimator_important_features['importances_std'] < .2)[0]))))

    print("features_indices:", features_indices)

    important_features_names = list(pld_complete.columns[features_indices])
    important_features_values = estimator_important_features['importances_mean'][features_indices].reshape(1, -1)

    indices = np.argsort(important_features_values)[0][::-1].tolist()

    print("Permutation important Features w.r.t " + name + " Reg.:",)
    #     for i in indices:
    #         print("%.3f" % important_features_values[0][i], important_features_names[i], )

    return indices, important_features_values, important_features_names


def apply_shap_summary_plot(model, x_train, y_train, x_test, y_test,
                            model_name, n_clusters, data_type, ):
    # Model-Agnostic Approximations (4.1) >> Linear LIME + Shapley values
    explainer = shap.KernelExplainer(model.predict, shap.kmeans(x_train, n_clusters))

    # Explain all the predictions in the test set
    plt.figure(figsize=(25, 15))
    plt.subplot(1, 1, 1)
    shap_values = explainer.shap_values(x_test)

    if model_name == "RF" or model_name == "GBR":
        values = np.abs(shap_values).mean(0)  # bcz it is list of matrices
        indices = np.argsort(values)[::-1].tolist()
        names = x_train.columns.tolist()

    else:

        print('other models shap values:', type(shap_values), len(shap_values), shap_values[0].shape)
        print(" ")

        values = np.abs(shap_values[0]).mean(0)  # bcz it is list of matrices
        indices = np.argsort(values)[::-1].tolist()
        names = x_train.columns.tolist()

        # print("names:", names)
        # print("values:", values)
        # print("indices:", indices)

        if "Ic_norm" in names:
            idx = names.index("Ic_norm")
            names.remove("Ic_norm")
            indices.pop(idx)
            values = np.delete(values, idx)

        # print("names:", names)
        # print("values:", values)
        # print("indices:", indices)

    values = [values]  # list of values for the sake of compatibility

    shap.summary_plot(shap_values, x_test, plot_type="bar", show=False)
    plt.title("{} on {} Summary bar plot".format(model_name, data_type), )
    plt.xlabel('mean(|SHAP values|) (average impact on model output magnitude)', fontsize=12)
    plt.savefig("shap_figs/summary bar plot of " + model_name + " on " + data_type + " data.png",
                bbox_inches='tight')
    plt.figure(figsize=(25, 15))
    plt.subplot(1, 1, 1)
    shap.summary_plot(shap_values, x_test, show=False)
    plt.title("{} on {} Summary scatter plot".format(model_name, data_type), )
    plt.xlabel('mean(|SHAP values|) (average impact on model output)', fontsize=12)

    plt.subplots_adjust(hspace=.65, wspace=.45)
    plt.savefig("shap_figs/summary scatter plot of " + model_name + " on " + data_type + " data.png",
                bbox_inches='tight')

    features_importance_shap = (indices, values, names)

    return explainer, features_importance_shap

