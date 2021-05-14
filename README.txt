# Similarity Analysis of Genomic SequencesUsing Convex Hull in Hamming Space

Zülal Bingöl - 21301083 - zulal.bingol@bilkent.edu.tr
Ricardo Román-Brenes - 22001125 - ricardo@bilkent.edu.tr




In this project, we  developed a procedure for similarity comparison and clustering of genomic sequences using convex hulls in Hamming space. Our project is based on the research done by Campo and Khudyakov in 2020., which uses a novel clustering algorithm, *k-hulls*, for improving the Convex Hull Distance algorithm on heterogeneous data.



## Install requirements

pip install -r requirements.txt


## How to run


The program is made up for a 2D version and a 3D version of the convex hull generator. The 3D version is meant for comparisons purposes only and was not developed by us.

### 2D Convex Hull

In the CH_2D directory:

python main.py [DATAFILE] [NUMBER OF HULLS]


For example:

python main.py shortreads.fasta 4


This will produce a 2D scatter plot visualization of the hulls that the user can manipulate.

### 3D Convex Hull

In the CH_3D directory:

python main.py [DATAFILE] [NUMBER OF HULLS]


For example:

python main.py shortreads.fasta 4


This will open a new tab or window in the default web browser with a 3D surface plot that the user can manipulate.

# Main Reference
D. S. Campo and Y. Khudyakov, “Convex hulls in hamming space enable efficient searchfor similarity and clustering of genomic sequences”, BMC bioinformatics, vol. 21, no. 18,pp. 1–13, 2020