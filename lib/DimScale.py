#! python3

from matplotlib.pyplot import contour
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import plotly.graph_objects as go
import itertools, math, numpy as np

from scipy.spatial import ConvexHull


def PCA_2(data):

    X = data.iloc[:, 0:-1]

    pca = PCA(n_components=2)
    components = pca.fit_transform(X)

    fig_1 = px.scatter(
        components, x=0, y=1, color=data[data.columns[-1]], title="Reduction to 2 PCA"
    )
    # fig_1.show()

    total_var = pca.explained_variance_ratio_.sum() * 100
    print("Variance explained with 2 PCAs: ", total_var)

    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
    fig_2 = px.line(
        x=range(1, exp_var_cumul.shape[0] + 1),
        y=exp_var_cumul,
        # mode='lines+markers',
        labels={"x": "# Components", "y": "Explained Variance"},
        title="Variance explained with 2 PCAs: " + str(total_var),
    )
    # fig_2.show()

    return {
        "fig_pca": fig_1,
        "fig_exp_var": fig_2,
        "total_variance": total_var,
        "pca": components,
    }


def PCA_3(data):

    X = data.iloc[:, 0:-1]
    pca = PCA(n_components=3)
    components = pca.fit_transform(X)

    fig_1 = px.scatter_3d(
        components,
        x=0,
        y=1,
        z=2,
        color=data[data.columns[-1]],
        title="Reduction to 3 PCA",
    )
    # fig_1.show()

    total_var = pca.explained_variance_ratio_.sum() * 100
    print("Variance explained with 3 PCAs: ", total_var)

    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
    fig_2 = px.line(
        x=range(1, exp_var_cumul.shape[0] + 1),
        y=exp_var_cumul,
        # mode='lines+markers',
        labels={"x": "# Components", "y": "Explained Variance"},
        title="Variance explained with 3 PCAs: " + str(total_var),
    )
    # fig_2.show()

    return {
        "fig_pca": fig_1,
        "fig_exp_var": fig_2,
        "total_variance": total_var,
        "pca": components,
    }



def visualize_ch3d(ch, data):
    
    
    ch = data[ch.vertices]
    
    fig = go.Figure()

    fig.add_trace(
        go.Mesh3d(
            x=ch[:,0], 
            y=ch[:,1], 
            z=ch[:,2], 
            opacity=0.5, 
            alphahull=0,
            contour=go.mesh3d.Contour(color="black", show=True, width=2),
            
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=ch[:,0], 
            y=ch[:,1], 
            z=ch[:,2], 
            mode="markers"
        )
    )

    fig.show()


    return


def visualize_3d(pca):
    x = np.array(pca)
    ch = x[ConvexHull(x).vertices]
    print("points of data", len(x))
    print("points in hull", len(ch))

    fig = go.Figure()

    fig.add_trace(
        go.Mesh3d(
            x=ch[:, 0], 
            y=ch[:, 1], 
            z=ch[:, 2], 
            color="blue", 
            opacity=0.6, 
            alphahull=0
        )
    )

    fig.add_trace(
        go.Scatter3d(x=x[:, 0], 
        y=x[:, 1], 
        z=x[:, 2], 
        mode="markers")
    )

    fig.show()
    return