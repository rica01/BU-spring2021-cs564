#! python3


import sys
sys.path.insert(0, '../lib')
from dna2vec import *
from DimScale import *
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from pprint import pprint
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# data should be presented in a Pandas DataFrame with all numerical 
# attributes but the last one (category/class)
reads = []
l = 200

for i in range(l):
    reads.append(dna2vec(random_dna_seq(l)))



df = pd.DataFrame(reads)
pprint(df)

r = PCA_3(df)

#r['fig_pca'].show()
vec_save(r['pca'], 'out.vec')
# pprint(r)

ch3d = ConvexHull(r["pca"])
visualize_ch3d(ch3d, r["pca"])
# visualize_3d(r['pca'])




