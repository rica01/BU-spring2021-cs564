#! python3
import sys
sys.path.insert(0, '../lib')

from KHulls import KHulls
from GrahamsScan import GrahamsScan

from dna2vec import *
from DimScale import *
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import csv
from pandas import DataFrame


# Temporary patch unaligned reads with different lengths
def handle_diff_rl(org_rl, new_rl, read):
    if new_rl < org_rl:
        for i in range(new_rl, org_rl):
            read = read + "-"
    else:
        suffix = new_rl - org_rl
        read = read[:-suffix]
    return read


def read_fasta_file(inputfile):
    file1 = open(inputfile, 'r')
    lines = file1.readlines()

    namestags = []
    reads = []
    count = 0
    read_length = 0
    for line in lines:
        if count % 2 == 0:
            tag = line.strip()
            namestags.append(tag)
        elif count % 2 != 0:
            read = line.strip()
            if read_length == 0:
                read_length = len(read)
            elif len(read) != read_length:
                #print("Warning(line", count+1,"): Different read length recognized:", read_length, "--", len(read))
                read = handle_diff_rl(read_length, len(read), read)
                #print("Warning: New read length: ", len(read))
            reads.append(read)
        count += 1

    file1.close()
    return reads, namestags


# Returns the lists of reads and labels.
# dataset-specific for Hepatitis C data from D.C.
def get_HC_reads(inputfile):
    file1 = open(inputfile, 'r')
    lines = file1.readlines()

    reads = []
    tags = []

    read_length = -1
    count = 0
    read = ""
    for line in lines:
        if count == 0:
            label = line.strip()
            tags.append(label[1])
        elif count <= 4:
            subread = line.strip()
            read = read + subread
            #print("'", read, "'--", read[0], "--")
            if count == 4:
                if read_length == -1:
                    read_length = len(read)
                    reads.append(read)
                elif len(read) == read_length:
                    reads.append(read)
                else:
                    tags.pop(1)
                read = ""
                count = -1
        count += 1
    return reads, tags


def main():

    data_file = sys.argv[1]
    #reads, labels = get_HC_reads(data_file)
    reads, labels = read_fasta_file(data_file)

    encodings = []
    for i in range(len(reads)):
        encodings.append(dna2vec(reads[i]))

    df_encodings = pd.DataFrame(encodings)
    r = PCA_2(df_encodings)
    points = r["pca"]
    reads = np.array(reads)
    khulls = KHulls(int(sys.argv[2]), points, reads)
    khulls.run_CH_based()

    '''
    # TODO: The dataset should be organized to be able to extract the true labels for SARS-COV-2.
    # in the meantime, colorings should the cluster assignments by the algorithm
    # labels = pangolin_lineage
    # inforfile = argv[2]
    info_df = pd.read_csv(infofile, sep="\t")
    #print(info_df['pangolin_lineage'])
    print("num clusters = ", info_df['pangolin_lineage'].nunique())
    label_list = info_df.pangolin_lineage.unique()
    print(label_list)
    print(label_list.shape[0])
    sys.exit()
    '''

    # Forming individual convex hulls --> TODO: define color palette for k > 6
    colors = ['c-', 'm-', 'y-', 'k-', 'r-', 'b-']
    for i in range(0, khulls.k):
        khulls.member_groups[i] = np.array(khulls.member_groups[i])
        temp_graham=GrahamsScan(khulls.member_groups[i])
        temp_convex_hull = temp_graham.run()
        filename = "ch_" + str(i)
        temp_graham.ch_plot(temp_convex_hull, filename, colors[i])

    # Plotting
    plt.scatter(khulls.points[:, 0], khulls.points[:, 1],
                c=khulls.memberships, s=20, cmap='viridis')
    
    # annotating the points
    #for i, txt in enumerate(labels):
        #plt.annotate(txt, (self.points[i][0], self.points[i][1]))

    #plt.plot(self.points[seeds[0]][0], self.points[seeds[0]][1], 'r*')
    #plt.plot(self.points[seeds[1]][0], self.points[seeds[1]][1], 'r*')
    #plt.plot(self.points[seeds[2]][0], self.points[seeds[2]][1], 'r*')
    #plt.plot(khulls.points[-1][0], khulls.points[-1][1], 'r*') # added this line to see Wuhan variant
    plt.show()


if __name__ == "__main__":
    main()
