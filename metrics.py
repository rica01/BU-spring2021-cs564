import numpy as np
from dna2vec import *

characters = ['A', 'C', 'G', 'T', '*']

read_length = 5
reads = []
reads_2 = []

reads.append('AAAAG')
reads.append('AAACG')
reads.append('AAAGG')
reads.append('TAAAG')

reads_2.append('AAAAG')
reads_2.append('AAACG')
reads_2.append('AAAGG')
reads_2.append('TAAAG')

reads = np.array(reads)
reads_2 = np.array(reads_2)

def encode(character):
    if character == 'A': return 0
    elif character == 'C': return 1
    elif character == 'G': return 2
    elif character == 'T': return 3
    else: return 4


def calc_ch_str(reads, read_length, num_reads):
    ch_str = ""
    for i in range(0, read_length):
        current_ch = reads[0][i]
        j = 1
        while (j < num_reads and reads[j][i] == current_ch):
            j = j + 1
        if (j < num_reads):
            ch_str = ch_str + "*"
        else:
            ch_str = ch_str + current_ch
    return ch_str


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
        j = 0
        k = 1
        while (j < len(characters)-1):
            inner_sum = inner_sum + freqs[p][j] * freqs[p][k]
            k = k + 1
            if k == len(characters):
                j = j + 1 
                k = j + 1
        outer_sum = outer_sum + inner_sum
    
    denom = ((num_reads ** 2) - num_reads) / 2
    return outer_sum / denom


#avg distance between populations --> this func makes no sense
#the number of reads should be equal
def avg_distance_pops(freqs_p, freqs_q, read_length, num_reads):
    outer_sum = 0
    for p in range(0, read_length):  # position
        inner_sum = 0
        j = 0
        k = 1
        while (j < len(characters)-1):
            inner_sum = inner_sum + freqs_p[p][j] * freqs_q[p][k]
            k = k + 1
            if k == len(characters):
                j = j + 1
                k = j + 1
        outer_sum = outer_sum + inner_sum
    return outer_sum / (num_reads ** 2)


def ch_str_distance(str_reads_1, str_reads_2, read_length):
    ch_str_dist = 0
    for p in range(0, read_length):  # position
        if (str_reads_1[p] != '*') and (str_reads_1[p] != str_reads_2[p]):
            ch_str_dist = ch_str_dist + 1
    return ch_str_dist


ch_str_1 = calc_ch_str(reads, read_length, reads.shape[0])
ch_str_2 = calc_ch_str(reads_2, read_length, reads_2.shape[0])
ch_str_dist = ch_str_distance(ch_str_1, ch_str_2, read_length)
print("ch_str_1 =", ch_str_1)
print("ch_str_2 =", ch_str_2)
print("ch_str_dist = ", ch_str_dist)


#print(reads)
freqs = frequency(reads, read_length)
freqs_2 = frequency(reads_2, read_length)
#print(freq)

cc = closeness_centrality(freqs, read_length, 'TAAAG')
print("cc = ", cc)

ahd = avg_hamming_dist(freqs, read_length, reads.shape[0])
print("ahd =", ahd)

adp = avg_distance_pops(freqs, freqs_2, read_length, reads.shape[0])
print("adp =", adp)
