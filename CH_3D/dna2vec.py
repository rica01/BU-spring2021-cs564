import numpy as np
from datetime import datetime


# dna_string is a combination of ACGT characters
def dna2vec(dna_string):
    vec_len = len(dna_string)
    vec_dna = np.ndarray(shape=(1, vec_len))
    vec_dna = {}
    # print(dna_string)

    for b in range(vec_len):
        if dna_string[b] == 'A' or dna_string[b] == 'a':
            vec_dna[b+1] = 0
        elif dna_string[b] == 'C' or dna_string[b] == 'c':
            vec_dna[b+1] = 1
        elif dna_string[b] == 'G' or dna_string[b] == 'g':
            vec_dna[b+1] = 2
        elif dna_string[b] == 'T' or dna_string[b] == 't':
            vec_dna[b+1] = 3
        else:
            vec_dna[b+1] = -1

    vec_dna["sp"] = dna_string.__hash__()
    # print(vec_dna)
    return vec_dna


def vec_save(vectors, filename):
    with open(filename, 'w') as f:
        f.write(str(len(vectors))+'\n')
        for v in vectors:
            f.write(str(v[0]) + ' ' + str(v[1]) + ' ' + str(v[2]) + '\n')
    return


def random_dna_seq(length):
    s = ''
    for i in range(length):
        b = np.random.randint(-1, 4)
        if b == -1:
            s += '*'
        elif b == 0:
            s += 'a'
        elif b == 1:
            s += 'c'
        elif b == 2:
            s += 'g'
        elif b == 3:
            s += 't'
    return s