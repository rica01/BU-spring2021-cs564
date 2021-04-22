import numpy as np
import pandas as pd
from GrahamsScan import GrahamsScan
from metrics import *

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

    def run(self):
        read_length = len(self.reads[0])
        num_reads = self.reads.shape[0]
        ch_str = calc_ch_str(self.reads, read_length, num_reads)
        freqs = frequency(self.reads, read_length)
        
        # closeness centrality of sequences
        min_cc = closeness_centrality(freqs, read_length, self.reads[0])
        first_seed_id = 0
        second_seed_id = 0
        second_min_cc = 0
        for i in range(1, num_reads):
            cc = closeness_centrality(freqs, read_length, self.reads[i])
            if (i == 1):
                if (cc < min_cc):
                    second_min_cc = min_cc
                    second_seed_id = 0
                    min_cc = cc
                    first_seed_id = 1
                else:
                    second_min_cc = cc
                    second_seed_id = 1
            elif (i > 1) and (cc < min_cc):
                second_min_cc = min_cc
                second_seed_id = first_seed_id
                min_cc = cc
                first_seed_id = i
            elif (i > 1) and (cc > min_cc) and (cc < second_min_cc):
                second_min_cc = cc
                second_seed_id = i
            print("cc ", i, cc)
        
        print("ch_str = ", ch_str)
        print("min_cc = ", first_seed_id, min_cc)
        print("second min_cc = ", second_seed_id, second_min_cc)
        




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
    num_reads = 10

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
    khulls.run()



if __name__ == "__main__":
    main()
