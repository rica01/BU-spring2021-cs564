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
from GrahamsScan import GrahamsScan




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
visualize_3d(r['pca'])


#Graham's Scan
r = PCA_2(df)
points = r["pca"]
graham = GrahamsScan(points)
convex_hull = graham.run()
graham.ch_plot_show(convex_hull, "graham")



'''
pprint(r["pca"])

points = r["pca"]
hull = ConvexHull(points)
#Plot it:


plt.plot(points[:,0], points[:,1], 'o')
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
#We could also have directly used the vertices of the hull, which for 2-D are guaranteed to be in counterclockwise order:


plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
plt.show()
'''
