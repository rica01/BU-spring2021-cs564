import sys
sys.path.insert(0, '../lib')


import numpy as np
import pandas as pd
from GrahamsScan import GrahamsScan
from metrics import *


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from DimScale import *
from dna2vec import *


class KHulls:

    def __init__(self, k, C, points, reads):
        self.k = k  #number of clusters
        self.clusters = C #cluster memberships 
        self.points = points
        self.reads = reads
    
    def set_k(k):
        self.k = k

    def run_str_based(self):
        read_length = len(self.reads[0])
        num_reads = self.reads.shape[0]
        memberships = np.full(num_reads, -1)
        seeds = np.full(self.k, -1)
        
        ch_str = calc_ch_str(self.reads, read_length, num_reads)
        freqs = frequency(self.reads, read_length)
        
        # closeness centrality of sequences
        min_cc = closeness_centrality(freqs, read_length, self.reads[0])
        seeds[0] = 0 #first seed
        for i in range(1, num_reads):
            cc = closeness_centrality(freqs, read_length, self.reads[i])
            if cc < min_cc:
                min_cc = cc
                seeds[0] = i
        
        seeds[1] = 0 # second seed
        ch_str_dist = 0
        for i in range(0, num_reads):
            if i != seeds[0]:
                temp_dist = ch_str_distance(
                    self.reads[seeds[0]], self.reads[i], read_length)
                if (ch_str_dist < temp_dist):
                    ch_str_dist = temp_dist
                    seeds[1] = i

        memberships[seeds[0]] = seeds[0]
        memberships[seeds[1]] = seeds[1]

        if (self.k > 2): # didn't check this part, be cautious!
            i = 3
            while i <= self.k:
                max_geo_mean = 1
                max_geo_mean_id = 0

                for t in range(0, num_reads):
                    if memberships[t] == -1:
                        z = 0
                        temp_geo = 1
                        while seeds[z] != -1:
                            temp_geo = temp_geo * ch_str_distance(self.reads[seeds[z]], self.reads[t], read_length)
                            z = z + 1
                        temp_geo = temp_geo ** (1/(i-1))
                        if temp_geo > max_geo_mean:
                            max_geo_mean = temp_geo
                            max_geo_mean_id = t

                seeds[i] = max_geo_mean_id
                memberships[seeds[i]] = seeds[i]
                i = i + 1





        print("ch_str = ", ch_str)
        print("min_cc = ", seeds[0], min_cc)
        print("second seed_id = ", seeds[1])

        plt.plot(self.points[:, 0], self.points[:, 1], 'o')
        plt.plot(self.points[seeds[0]][0], self.points[seeds[0]][1], 'ro')
        plt.plot(self.points[seeds[1]][0], self.points[seeds[1]][1], 'rx')

        plt.show()

        




def main():

    '''
    reads = []
    reads.append('AAAAG')
    reads.append('AAACG')
    reads.append('AAAGG')
    reads.append('TAAAG')
    reads = np.array(reads)
    '''
    reads = []
    encodings = []
    read_length = 5
    num_reads = 30

    for i in range(num_reads):
        read = random_dna_seq(read_length)
        reads.append(read)
        encodings.append(dna2vec(read))

    df_encodings = pd.DataFrame(encodings)
    reads = np.array(reads)
    r = PCA_2(df_encodings)
    points = r["pca"]

    print(reads)

    clusters = []
    khulls = KHulls(2, clusters, points, reads)
    khulls.run_str_based()



if __name__ == "__main__":
    main()
