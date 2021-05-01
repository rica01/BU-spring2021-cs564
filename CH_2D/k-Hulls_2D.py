from shapely.geometry import Polygon
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
        
        print(memberships)
       
        while True:
            change = 0
            for i in range(0, num_reads):
                j = 0
                min_dist = read_length+1
                min_dist_id = memberships[i]
                while (j < self.k):
                    ch_str_dist = ch_str_distance(self.reads[i], self.reads[seeds[j]], read_length)
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

        print("ch_str = ", ch_str)
        print("min_cc = ", seeds[0], min_cc)
        print("second seed_id = ", seeds[1])

        '''
        for i in range(0, num_reads):
            if memberships[i] == seeds[0]:
                plt.plot(self.points[i][0], self.points[i][1], 'ro')
            else:
                plt.plot(self.points[i][0], self.points[i][1], 'rx')
        plt.scatter(
            self.points[seeds[0]][0], self.points[seeds[0]][1], c='black', s=200, alpha=0.5)
        plt.scatter(
            self.points[seeds[1]][0], self.points[seeds[1]][1], c='black', s=200, alpha=0.5)
        '''

        plt.scatter(self.points[:, 0], self.points[:, 1], c=memberships, s=50, cmap='viridis')
        #plt.plot(self.points[:, 0], self.points[:, 1], 'o')
        plt.plot(self.points[seeds[0]][0], self.points[seeds[0]][1], 'ro')
        plt.plot(self.points[seeds[1]][0], self.points[seeds[1]][1], 'rx')
        plt.show()

    def run_CH_based(self):
        read_length = len(self.reads[0])
        num_reads = self.reads.shape[0]
        memberships = np.full(num_reads, -1)
        seeds = np.full(self.k, -1) # contains ID's of seeds in points array

        graham_all = GrahamsScan(self.points)
        convex_hull = graham_all.run()

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

        plt.plot(x, y, 'rx') # red-x = mean of polygon coordinates

        # calculating centroid of polygon
        polygon = Polygon(convex_hull)
        print(polygon.centroid)
        plt.plot(polygon.centroid.x, polygon.centroid.y, 'bx') # blue-x = centroid

        # finding seeds distant to centroid
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

        #print("first seed = ", seeds[0], self.points[seeds[0]])
        #print("second seed = ", seeds[1], self.points[seeds[1]])

        while True: #TODO: some memberships remain -1, fix it
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

        #partitioning the point set
        subpoints = []
        for i in range(0, self.k):
            subpoints.append([])

        for i in range(0, num_reads):
            j = 0
            while j < self.k:
                if memberships[i] == seeds[j]:
                    subpoints[j].append(self.points[i])
                j = j + 1
        
        
        # Forming individual convex hulls --> TODO: define color palette for k > 2
        colors = ['co-', 'mo-', 'yo-']
        for i in range(0, self.k):
            subpoints[i] = np.array(subpoints[i])
            temp_graham = GrahamsScan(subpoints[i])
            temp_convex_hull = temp_graham.run()
            filename = "ch_" + str(i)
            temp_graham.ch_plot(temp_convex_hull, filename, colors[i])


        #plt.plot(self.points[:, 0], self.points[:, 1], 'o')
        plt.scatter(self.points[:, 0], self.points[:, 1],
                    c=memberships, s=50, cmap='viridis')
        plt.plot(self.points[seeds[0]][0], self.points[seeds[0]][1], 'cx')
        plt.plot(self.points[seeds[1]][0], self.points[seeds[1]][1], 'mx')
        plt.show()

        # TODO: approaches to try for possible biological meaning:
        # - use closeness_centrality instead of centroid
        # - incorporate hamming distance in proximity calculations
        # - 

       

def main():

    reads = []
    encodings = []
    read_length = 10
    num_reads = 100

    for i in range(num_reads):
        read = random_dna_seq(read_length)
        reads.append(read)
        encodings.append(dna2vec(read))

    df_encodings = pd.DataFrame(encodings)
    reads = np.array(reads)
    r = PCA_2(df_encodings)
    points = r["pca"]

    #print(reads)

    clusters = []
    khulls = KHulls(3, clusters, points, reads)
    #khulls.run_str_based()

    khulls.run_CH_based()


if __name__ == "__main__":
    main()
