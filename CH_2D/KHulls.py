import sys
sys.path.insert(0, '../lib')

from dna2vec import *
from DimScale import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from metrics import *
from GrahamsScan import GrahamsScan
import pandas as pd
import numpy as np
from shapely.geometry import Polygon


class KHulls:

    def __init__(self, k, points, reads):
        self.k = k  # number of clusters
        self.points = points
        self.reads = reads
        self.memberships = []
        self.member_groups = []

    def set_k(k):
        self.k = k

    def run_str_based(self, labels):
        read_length = len(self.reads[0])
        num_reads = self.reads.shape[0]
        memberships = np.full(num_reads, -1)
        seeds = np.full(self.k, -1)

        ch_str = calc_ch_str(self.reads, read_length, num_reads)
        freqs = frequency(self.reads, read_length)

        # closeness centrality of sequences
        min_cc = closeness_centrality(freqs, read_length, self.reads[0])
        seeds[0] = 0  # first seed
        for i in range(1, num_reads):
            cc = closeness_centrality(freqs, read_length, self.reads[i])
            if cc < min_cc:
                min_cc = cc
                seeds[0] = i

        seeds[1] = 0  # second seed
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

        if (self.k > 2):  # didn't check this part, be cautious!
            i = 2
            while i+1 <= self.k:
                max_geo_mean = 1
                max_geo_mean_id = 0

                for t in range(0, num_reads):
                    if memberships[t] == -1:
                        z = 0
                        temp_geo = 1
                        while seeds[z] != -1:
                            temp_geo = temp_geo * \
                                ch_str_distance(
                                    self.reads[seeds[z]], self.reads[t], read_length)
                            z = z + 1
                        temp_geo = temp_geo ** (1/i)
                        if temp_geo > max_geo_mean:
                            max_geo_mean = temp_geo
                            max_geo_mean_id = t

                seeds[i] = max_geo_mean_id
                memberships[seeds[i]] = seeds[i]
                i = i + 1

        while True:
            change = 0
            for i in range(0, num_reads):
                j = 0
                min_dist = read_length+1
                min_dist_id = memberships[i]
                while (j < self.k):
                    ch_str_dist = ch_str_distance(
                        self.reads[i], self.reads[seeds[j]], read_length)
                    if ch_str_dist < min_dist:
                        min_dist = ch_str_dist
                        min_dist_id = seeds[j]
                    j = j + 1

                if memberships[i] != min_dist_id:
                    memberships[i] = min_dist_id
                    change = 1

            if change == 0:
                break

        print(memberships)
        self.memberships = memberships

        plt.scatter(self.points[:, 0], self.points[:, 1],
                    c=memberships, s=50, cmap='viridis')
        for i, txt in enumerate(labels):
            plt.annotate(txt, (self.points[i][0], self.points[i][1]))
        plt.plot(self.points[seeds[0]][0], self.points[seeds[0]][1], 'ro')
        plt.plot(self.points[seeds[1]][0], self.points[seeds[1]][1], 'rx')
        plt.show()


    def run_CH_based(self):
        read_length = len(self.reads[0])
        num_reads = self.reads.shape[0]
        memberships = np.full(num_reads, -1)
        seeds = np.full(self.k, -1)  # contains ID's of seeds in points array

        graham_all = GrahamsScan(self.points)
        convex_hull = graham_all.run()
        #graham_all.ch_plot(convex_hull, "0_convex_hull", 'k-.')

        # Calculating mean of polygon coordinates
        i = 0
        x = 0
        y = 0
        while i < convex_hull.shape[0]:
            x = x + convex_hull[i][0]
            y = y + convex_hull[i][1]
            i = i + 1
        x = x / convex_hull.shape[0]
        y = y / convex_hull.shape[0]
        plt.plot(x, y, 'rx')  # blue-x = mean of polygon coordinates

        # calculating centroid of polygon
        polygon = Polygon(convex_hull)
        plt.plot(polygon.centroid.x, polygon.centroid.y,
                 'bx')  # red-x = centroid

        # finding seeds distant to polygon centroid
        max_distance = euclidean_dist(
            polygon.centroid.x, polygon.centroid.y, self.points[0][0], self.points[0][1])
        seeds[0] = 0  # first seed
        for i in range(1, num_reads):
            temp_dist = euclidean_dist(
                polygon.centroid.x, polygon.centroid.y, self.points[i][0], self.points[i][1])
            if temp_dist > max_distance:
                max_distance = temp_dist
                seeds[0] = i

        seeds[1] = 0  # second seed
        max_distance = 0
        for i in range(0, num_reads):
            if i != seeds[0]:
                temp_dist = euclidean_dist(
                    self.points[seeds[0]][0], self.points[seeds[0]][1], self.points[i][0], self.points[i][1])
                if (max_distance < temp_dist):
                    max_distance = temp_dist
                    seeds[1] = i

        memberships[seeds[0]] = seeds[0]
        memberships[seeds[1]] = seeds[1]

        # Additional seeds for k > 2
        if (self.k > 2):  # didn't check this part, be cautious!
            i = 2
            while i+1 <= self.k:
                max_geo_mean = 1
                max_geo_mean_id = 0

                for t in range(0, num_reads):
                    if memberships[t] == -1:
                        z = 0
                        temp_geo = 1
                        while seeds[z] != -1:
                            temp_geo = temp_geo * \
                                euclidean_dist(
                                    self.points[seeds[z]][0], self.points[seeds[z]][1], self.points[t][0], self.points[t][1])
                            z = z + 1
                        temp_geo = temp_geo ** (1/i)
                        if temp_geo > max_geo_mean:
                            max_geo_mean = temp_geo
                            max_geo_mean_id = t

                seeds[i] = max_geo_mean_id
                memberships[seeds[i]] = seeds[i]
                i = i + 1

        #print("seeds", seeds)
        #print(memberships)

        # Clustering
        while True:
            change = 0
            for i in range(0, num_reads):
                min_dist = euclidean_dist(
                    self.points[seeds[0]][0], self.points[seeds[0]][1], self.points[i][0], self.points[i][1])
                min_dist_id = seeds[0]
                j = 1
                while (j < self.k):
                    temp_dist = euclidean_dist(
                        self.points[seeds[j]][0], self.points[seeds[j]][1], self.points[i][0], self.points[i][1])
                    if temp_dist < min_dist:
                        min_dist = temp_dist
                        min_dist_id = seeds[j]
                    j = j + 1

                if memberships[i] != min_dist_id:
                    memberships[i] = min_dist_id
                    change = 1

            if change == 0:
                break

        print(memberships)
        self.memberships = memberships

        # Partitioning the point set
        subpoints = []
        for i in range(0, self.k):
            subpoints.append([])

        for i in range(0, num_reads):
            j = 0
            while j < self.k:
                if memberships[i] == seeds[j]:
                    subpoints[j].append(self.points[i])
                j = j + 1

        self.member_groups = subpoints

'''

def main():

    reads = []
    
    reads.append("AAAAACCCC")
    reads.append("ACAAACCCC")
    reads.append("AGAAACCCC")
    reads.append("ATAAACCCC")

    reads.append("AAAAAGGGG")
    reads.append("ACAAAGGGG")
    reads.append("AGAAAGGGG")
    reads.append("ATAAAGGGG")

    reads.append("AAAAATTTT")
    reads.append("ACAAATTTT")
    reads.append("AGAAATTTT")
    reads.append("ATAAATTTT")

    encodings = []
    read_length = 9
    labels = ['C', 'C', 'C', 'C', 'G', 'G', 'G', 'G', 'T', 'T', 'T', 'T']

    for i in range(len(reads)):
        encodings.append(dna2vec(reads[i]))

    df_encodings = pd.DataFrame(encodings)
    reads = np.array(reads)
    r = PCA_2(df_encodings)
    points = r["pca"]

    clusters = []
    khulls = KHulls(3, clusters, points, reads)
    
    #khulls.run_str_based()
    khulls.run_CH_based()

if __name__ == "__main__":
    main()

'''
