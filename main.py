#! python3
# pip install plo

import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from DimScale import *


# data should be presented in a Pandas DataFrame with all numerical 
# attributes but the last one (category/class)

# lists
A =     [  1,    1,    0,    0,    1,    0,    1,    1,    0,    0]
C =     [  0,    0,    1,    0,    1,    1,    1,    1,    1,    0]
G =     [  1,    0,    1,    0,    1,    0,    1,    0,    1,    0]
T =     [  1,    1,    1,    0,    1,    1,    1,    0,    0,    0]
CLS =   ["C1", "C1", "C2", "C3", "C3", "C4", "C5", "C6", "C7", "C7"]
  
# dictionary of lists 
dict = {'Adenine': A, 'Cytocine': C, 'Guanine': G, 'Tymine': T, 'class': CLS} 

df = pd.DataFrame(dict)

print(df)

########################### PCAs ###########################


r = PCA_2(df)
print(r)
r = PCA_3(df)
print(r)