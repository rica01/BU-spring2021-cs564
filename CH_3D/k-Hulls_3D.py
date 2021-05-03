
import sys

sys.path.insert(0, "../lib")

from pprint import pprint
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from dna2vec import *
from DimScale import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from metrics import *

# from GrahamsScan import GrahamsScan
import pandas as pd
import numpy as np
from shapely.geometry import Polygon


class KHulls:
    def __init__(self, k, C, points, reads):
        self.k = k  # number of clusters
        self.memberships = C
        self.points = points
        self.reads = reads

    def set_k(k):
        self.k = k

    def run_CH_based(self, labels):
        read_length = len(self.reads[0])
        num_reads = self.reads.shape[0]
        memberships = np.full(num_reads, -1)
        seeds = np.full(self.k, -1)  # contains ID's of seeds in points array

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
        plt.plot(x, y, "rx")  # red-x = mean of polygon coordinates

        # calculating centroid of polygon
        polygon = Polygon(convex_hull)
        plt.plot(polygon.centroid.x, polygon.centroid.y, "bx")  # blue-x = centroid

        # finding seeds distant to polygon centroid
        max_distance = euclidean_dist(
            polygon.centroid.x, polygon.centroid.y, self.points[0][0], self.points[0][1]
        )
        seeds[0] = 0  # first seed
        for i in range(1, num_reads):
            temp_dist = euclidean_dist(
                polygon.centroid.x,
                polygon.centroid.y,
                self.points[i][0],
                self.points[i][1],
            )
            if temp_dist > max_distance:
                max_distance = temp_dist
                seeds[0] = i

        seeds[1] = 0  # second seed
        max_distance = 0
        for i in range(0, num_reads):
            if i != seeds[0]:
                temp_dist = euclidean_dist(
                    self.points[seeds[0]][0],
                    self.points[seeds[0]][1],
                    self.points[i][0],
                    self.points[i][1],
                )
                if max_distance < temp_dist:
                    max_distance = temp_dist
                    seeds[1] = i

        memberships[seeds[0]] = seeds[0]
        memberships[seeds[1]] = seeds[1]

        # Additional seeds for k > 2
        if self.k > 2:  # didn't check this part, be cautious!
            i = 2
            while i + 1 <= self.k:
                max_geo_mean = 1
                max_geo_mean_id = 0

                for t in range(0, num_reads):
                    if memberships[t] == -1:
                        z = 0
                        temp_geo = 1
                        while seeds[z] != -1:
                            temp_geo = temp_geo * euclidean_dist(
                                self.points[seeds[z]][0],
                                self.points[seeds[z]][1],
                                self.points[t][0],
                                self.points[t][1],
                            )
                            z = z + 1
                        temp_geo = temp_geo ** (1 / i)
                        if temp_geo > max_geo_mean:
                            max_geo_mean = temp_geo
                            max_geo_mean_id = t

                seeds[i] = max_geo_mean_id
                memberships[seeds[i]] = seeds[i]
                i = i + 1

        # print("seeds", seeds)
        # print(memberships)

        # Clustering
        while True:
            change = 0
            for i in range(0, num_reads):
                min_dist = euclidean_dist(
                    self.points[seeds[0]][0],
                    self.points[seeds[0]][1],
                    self.points[i][0],
                    self.points[i][1],
                )
                min_dist_id = seeds[0]
                j = 1
                while j < self.k:
                    temp_dist = euclidean_dist(
                        self.points[seeds[j]][0],
                        self.points[seeds[j]][1],
                        self.points[i][0],
                        self.points[i][1],
                    )
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

        # Forming individual convex hulls --> TODO: define color palette for k > 4
        colors = ["c-", "m-", "y-", "k-", "r-", "b-"]
        for i in range(0, self.k):
            subpoints[i] = np.array(subpoints[i])
            temp_graham = GrahamsScan(subpoints[i])
            temp_convex_hull = temp_graham.run()
            filename = "ch_" + str(i)
            temp_graham.ch_plot(temp_convex_hull, filename, colors[i])

        # Plotting

        plt.scatter(
            self.points[:, 0], self.points[:, 1], c=memberships, s=20, cmap="viridis"
        )

        # annotating the points
        # for i, txt in enumerate(labels):
        # plt.annotate(txt, (self.points[i][0], self.points[i][1]))

        # plt.plot(self.points[seeds[0]][0], self.points[seeds[0]][1], 'cx')
        # plt.plot(self.points[seeds[1]][0], self.points[seeds[1]][1], 'mx')
        # plt.plot(self.points[seeds[2]][0], self.points[seeds[2]][1], 'rx')
        plt.legend()
        plt.show()

        # TODO: approaches to try for possible biological meaning:
        # - use closeness_centrality instead of centroid
        # - incorporate hamming distance in proximity calculations
        # -

    def run_CH_based_3D(self, labels):
        read_length = len(self.reads[0])
        num_reads = self.reads.shape[0]
        memberships = np.full(num_reads, -1)
        seeds = np.full(self.k, -1)  # contains ID's of seeds in points array

        c_h_3D = ConvexHull(self.points)
        convex_hull = c_h_3D.points

        # Calculating mean of polygon coordinates
        i = 0
        x = 0
        y = 0
        z = 0
        while i < convex_hull.shape[0]:
            x = x + convex_hull[i][0]
            y = y + convex_hull[i][1]
            z = z + convex_hull[i][2]
            i = i + 1
        x = x / convex_hull.shape[0]
        y = y / convex_hull.shape[0]
        z = z / convex_hull.shape[0]

        # fig = 
        fig = go.Figure()
        # ax = plt.axes(projection="3d")

        # ax.scatter(x, y, z, c="red", marker="x")  # red-x = mean of polygon coordinates
        fig.add_trace(
            go.Scatter3d(
                x=[x], 
                y=[y], 
                z=[z], 
                mode="markers"
            )
        )

        print("hull centroid:", x, y, z)

        # calculating centroid of polygon
        c_x = c_y = c_z = 0
        for p in convex_hull:
            c_x += p[0]
            c_y += p[1]
            c_z += p[2]
        c_x /= convex_hull.shape[0]
        c_y /= convex_hull.shape[0]
        c_z /= convex_hull.shape[0]

        # ax.scatter(c_x, c_y, c_z, c="blue", marker="x")  # blue-x = centroid
        fig.add_trace(
            go.Scatter3d(
                x=[c_x], 
                y=[c_y], 
                z=[c_z], 
                mode="markers"
            )
        )

        print("poly centroid:", c_x, c_y, c_z)



        # finding seeds distant to polygon centroid
        max_distance = euclidean_dist_3D(
            c_x, c_y, c_z, self.points[0][0], self.points[0][1], self.points[0][2]
        )
        seeds[0] = 0  # first seed
        for i in range(1, num_reads):
            temp_dist = euclidean_dist_3D(
                c_x,
                c_y,
                c_z,
                self.points[i][0],
                self.points[i][1],
                self.points[i][2],
            )
            if temp_dist > max_distance:
                max_distance = temp_dist
                seeds[0] = i

        seeds[1] = 0  # second seed
        max_distance = 0
        for i in range(0, num_reads):
            if i != seeds[0]:
                temp_dist = euclidean_dist_3D(
                    self.points[seeds[0]][0],
                    self.points[seeds[0]][1],
                    self.points[seeds[0]][2],
                    self.points[i][0],
                    self.points[i][1],
                    self.points[i][2],
                )
                if max_distance < temp_dist:
                    max_distance = temp_dist
                    seeds[1] = i

        memberships[seeds[0]] = seeds[0]
        memberships[seeds[1]] = seeds[1]

        # # Additional seeds for k > 2
        if self.k > 2:  # didn't check this part, be cautious!
            i = 2
            while i + 1 <= self.k:
                max_geo_mean = 1
                max_geo_mean_id = 0

                for t in range(0, num_reads):
                    if memberships[t] == -1:
                        z = 0
                        temp_geo = 1
                        while seeds[z] != -1:
                            temp_geo = temp_geo * euclidean_dist_3D(
                                self.points[seeds[z]][0],
                                self.points[seeds[z]][1],
                                self.points[seeds[z]][2],
                                self.points[t][0],
                                self.points[t][1],
                                self.points[t][2],
                            )
                            z = z + 1
                        temp_geo = temp_geo ** (1 / i)
                        if temp_geo > max_geo_mean:
                            max_geo_mean = temp_geo
                            max_geo_mean_id = t

                seeds[i] = max_geo_mean_id
                memberships[seeds[i]] = seeds[i]
                i = i + 1

        # #print("seeds", seeds)
        # # print(memberships)

        # Clustering
        while True:
            change = 0
            for i in range(0, num_reads):
                min_dist = euclidean_dist_3D(
                    self.points[seeds[0]][0],
                    self.points[seeds[0]][1],
                    self.points[seeds[0]][2],
                    self.points[i][0],
                    self.points[i][1],
                    self.points[i][2],
                )
                min_dist_id = seeds[0]
                j = 1
                while j < self.k:
                    temp_dist = euclidean_dist_3D(
                        self.points[seeds[j]][0],
                        self.points[seeds[j]][1],
                        self.points[seeds[j]][2],
                        self.points[i][0],
                        self.points[i][1],
                        self.points[i][2],
                    )
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

        # # Partitioning the point set
        subpoints = []
        for i in range(0, self.k):
            subpoints.append([])

        for i in range(0, num_reads):
            j = 0
            while j < self.k:
                if memberships[i] == seeds[j]:
                    subpoints[j].append(self.points[i])
                j = j + 1

        # # Forming individual convex hulls --> TODO: define color palette for k > 4
        colors = ["c-", "m-", "y-", "k-", "r-", "b-"]
        colors2 = ["cyan", "magenta", "yellow", "black", "red", "blue"]
        for i in range(0, self.k):
            subpoints[i] = np.array(subpoints[i])
            pprint(subpoints[i])
            temp_convex_hull_obj = ConvexHull(subpoints[i])
            temp_convex_hull = temp_convex_hull_obj.points

            fig.add_trace(
                go.Mesh3d(
                    x=temp_convex_hull[:, 0], 
                    y=temp_convex_hull[:, 1], 
                    z=temp_convex_hull[:, 2], 
                    color=colors2[i], 
                    opacity=0.6, 
                    alphahull=0
                )
            )
            # filename = "ch_" + str(i)
            # # temp_graham.ch_plot(temp_convex_hull, filename, colors[i])

            # n = temp_convex_hull.shape[0]
            # for k in range(0, n - 1):
            #     x = [temp_convex_hull[k][0], temp_convex_hull[k + 1][0]]
            #     y = [temp_convex_hull[k][1], temp_convex_hull[k + 1][1]]
            #     z = [temp_convex_hull[k][2], temp_convex_hull[k + 1][2]]
            #     # ax.plot(x, y, z, colors[i])
            #     # print(temp_convex_hull[i:i+2][0], temp_convex_hull[i:i+2][1])

            # # Last line connecting index 0 to n-1
            # x = [temp_convex_hull[0][0], temp_convex_hull[n - 1][0]]
            # y = [temp_convex_hull[0][1], temp_convex_hull[n - 1][1]]
            # z = [temp_convex_hull[0][2], temp_convex_hull[n - 1][2]]
            # # print(x, y)
            # # ax.plot(x, y, z, colors[i])

        # plt.scatter(self.points[:, 0], self.points[:, 1],
        #             c=memberships, s=20, cmap='viridis')


        
        # plt.legend()
        fig.show()

        return


def get_HC_reads(inputfile):
    file1 = open(inputfile, "r")
    lines = file1.readlines()

    reads = []
    labels = []

    read_length = -1
    count = 0
    read = ""
    for line in lines:
        if count == 0:
            label = line.strip()
            labels.append(label[1])
        elif count <= 4:
            subread = line.strip()
            read = read + subread
            # print("'", read, "'--", read[0], "--")
            if count == 4:
                if read_length == -1:
                    read_length = len(read)
                    reads.append(read)
                elif len(read) == read_length:
                    reads.append(read)
                else:
                    labels.pop(1)
                read = ""
                count = -1
        count += 1

    return reads, labels


def get_sim_reads(inputfile):
    file1 = open(inputfile, 'r')
    lines = file1.readlines()

    labels = []
    reads = []
    count = 0
    for line in lines:
        if count % 2 == 1:
            reads.append(line.strip())
        count += 1

    return reads, labels



def main():

    data_file = sys.argv[1]

    # reads, labels = get_HC_reads(data_file)
    reads, labels = get_sim_reads(data_file)
    encodings = []

    for i in range(len(reads)):
        encodings.append(dna2vec(reads[i]))

    df_encodings = pd.DataFrame(encodings)

    r = PCA_3(df_encodings)
    points = r["pca"]
    memberships = []
    reads = np.array(reads)
    khulls = KHulls(6, memberships, points, reads)
    khulls.run_CH_based_3D(labels)


if __name__ == "__main__":
    main()
