import numpy as np
from dna2vec import *

characters = ['A', 'C', 'G', 'T', '*']

read_length = 5
reads = []

#for i in range(0,num_reads):
#    read = random_dna_seq(read_length)
#    reads.append(read)
reads.append('AAAAG')
reads.append('AAACG')
reads.append('AAAGG')
reads.append('TAAAG')

reads = np.array(reads)

def encode(character):
    if character == 'A': return 0
    elif character == 'C': return 1
    elif character == 'G': return 2
    elif character == 'T': return 3
    else: return 4


def frequency(reads, read_lenth):
    freq = np.zeros([read_length, len(characters)])
    
    counts = np.zeros([len(characters)])
    for i in range (0, read_length):
        for j in range(0, reads.shape[0]):
            counts[encode(reads[j][i])] = counts[encode(reads[j][i])] + 1
        for j in range(0, len(characters)):
            freq[i][j] = counts[j] / reads.shape[0]
        counts.fill(0)
    return freq


def closeness_centrality(freqs, read_length, sequence):
    cc = 1
    for i in range(0, read_length):
        f = freqs[i][encode(sequence[i])]
        cc = cc * f
    return cc    

def avg_hamming_dist(freqs, read_length, num_reads):
    outer_sum = 0
    for p in range(0, read_length): #position
        inner_sum = 0
        product = 1
        t = 0
        for j in range(0, len(characters)):
            if t != j:
                product = freqs[p][j] * product
                inner_sum = inner_sum + product
            t = t + 1   


print(reads)
freq = frequency(reads, read_length)
print(freq)

cc = closeness_centrality(freq, read_length, 'TAAAG')
print(cc)