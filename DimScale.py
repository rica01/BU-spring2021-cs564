#! python3
# pip install plotly numpy scipy sklearn

import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def PCA_2(data):

    X = data.iloc[:, 0:-1]

    pca = PCA(n_components=2)
    components = pca.fit_transform(X)

    fig_1 = px.scatter(components,
                       x=0,
                       y=1,
                       color=data[data.columns[-1]],
                       title="Reduction to 2 PCA")
    fig_1.show()

    total_var = pca.explained_variance_ratio_.sum() * 100
    print("Variance explained with 2 PCAs: ", total_var)

    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
    fig_2 = px.line(
        x=range(1, exp_var_cumul.shape[0] + 1),
        y=exp_var_cumul,
        # mode='lines+markers',
        labels={
            "x": "# Components",
            "y": "Explained Variance"
        },
        title="Variance explained with 2 PCAs: " + str(total_var))
    fig_2.show()

    return {
        "fig_pca": fig_1,
        "fig_exp_var": fig_2,
        "total_variance": total_var
    }


def PCA_3(data):

    X = data.iloc[:, 0:-1]
    pca = PCA(n_components=3)
    components = pca.fit_transform(X)

    fig_1 = px.scatter_3d(components,
                          x=0,
                          y=1,
                          z=2,
                          color=data[data.columns[-1]],
                          title="Reduction to 3 PCA")
    fig_1.show()

    total_var = pca.explained_variance_ratio_.sum() * 100
    print("Variance explained with 3 PCAs: ", total_var)

    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
    fig_2 = px.line(
        x=range(1, exp_var_cumul.shape[0] + 1),
        y=exp_var_cumul,
        # mode='lines+markers',
        labels={
            "x": "# Components",
            "y": "Explained Variance"
        },
        title="Variance explained with 3 PCAs: " + str(total_var))
    fig_2.show()

    return {
        "fig_pca": fig_1,
        "fig_exp_var": fig_2,
        "total_variance": total_var
    }