import matplotlib.pyplot as plt
import sys
import numpy as np
from numpy import exp, abs

DEL = 1  # 0 = not deleted


class GrahamsScan:

    def __init__(self, points):
        self.points = points 
        self.delete = np.zeros(shape=(points.shape[0], 1))

    def getSize(self):
        print("delete is", self.delete.shape[0])

    def ch_plot_show(self, ch_set, filename):
        plt.plot(self.points[:, 0], self.points[:, 1], 'o')
        n = ch_set.shape[0]
        for i in range(0, n-1):
            x = [ch_set[i][0], ch_set[i+1][0]]
            y = [ch_set[i][1], ch_set[i+1][1]]
            plt.plot(x, y, 'ro-')
            #print(ch_set[i:i+2][0], ch_set[i:i+2][1])

        # Last line connecting index 0 to n-1
        x = [ch_set[0][0], ch_set[n-1][0]]
        y = [ch_set[0][1], ch_set[n-1][1]]
        #print(x, y)
        plt.plot(x, y, 'ro-')

        plt.savefig(filename)
        plt.show()


    def plot_show_save(self, points, filename):
        plt.plot(self.points[:, 0], self.points[:, 1], 'o')
        plt.savefig(filename)
        plt.show()


    def partition(self, subpoints, low, high):
        i = (low - 1)         # index of smaller element
        pivot = subpoints[high]

        for j in range(low, high):

            # subpoints[j] <= pivot:
            if (self.compare(subpoints[j], pivot, j, high) <= 0):
                i = i + 1
                subpoints[i][0], subpoints[j][0] = subpoints[j][0], subpoints[i][0]
                subpoints[i][1], subpoints[j][1] = subpoints[j][1], subpoints[i][1]

        subpoints[i+1][0], subpoints[high][0] = subpoints[high][0], subpoints[i+1][0]
        subpoints[i+1][1], subpoints[high][1] = subpoints[high][1], subpoints[i+1][1]
        return (i + 1)


    def quickSort(self, subpoints, low, high):
        if subpoints.shape[0] == 1:
            return subpoints

        if (low < high):
            pi = self.partition(subpoints, low, high)
            self.quickSort(subpoints, low, pi-1)
            self.quickSort(subpoints, pi+1, high)


    # Code 3.5: O'Rourke (82)
    # points should be global
    def compare(self, p_i, p_j, p_i_index, p_j_index):
        area = self.calc_area(self.points[0], p_i, p_j)  # pi < pj if Area2(p0, pi, pj) > 0
        if area > 0:
            return -1
        elif area < 0:
            return 1
        else:  # collinear
            x = abs(p_i[0] - self.points[0][0]) - abs(p_j[0] - self.points[0][0])
            y = abs(p_i[1] - self.points[0][1]) - abs(p_j[1] - self.points[0][1])

            if (x < 0) or (y < 0):
                self.delete[p_i_index] = DEL
                return -1
            elif (x > 0) or (y > 0):
                self.delete[p_j_index] = DEL
                return 1
            else:  # points are coincident
                if (p_i_index > p_j_index):
                    self.delete[p_j_index] = DEL
                else:
                    self.delete[p_i_index] = DEL
                return 0


    # Area2: a,b,c are points
    def calc_area(self, a, b, c):
        area = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
        return area

    # Left: Code 1.6 : O'Rourke (p29)


    def Left(self, a, b, c):
        return self.calc_area(a, b, c) > 0


    # Code 3.4: O'Rourke (p79)
    def find_lowest(self):
        m = 0  # lowest so far
        for i in range(1, self.points.shape[0]):
            if ((self.points[i][1] < self.points[m][1]) or ((self.points[i][1] == self.points[m][1]) and (self.points[i][0] > self.points[m][0]))):
                m = i
        # swapping points
        self.points[0][0], self.points[m][0] = self.points[m][0], self.points[0][0]
        self.points[0][1], self.points[m][1] = self.points[m][1], self.points[0][1]


    # Code 3.8 :  O'Rourke (p84)
    # difference = this creates a new set, rather than changing the original point set
    def squash(self):
        i = 0
        j = 0
        convex_set = np.zeros(shape=(self.points.shape[0], 2))
        while (i < self.points.shape[0]):
            if (self.delete[i] != DEL):
                convex_set[j][0] = self.points[i][0]
                convex_set[j][1] = self.points[i][1]
                j = j + 1
            i = i + 1
        convex_set = convex_set[0:j]
        return convex_set


    def graham_scan(self, convex_set):

        #top element = stack[len(stack)-1]
        #top -> next = stack[len(stack)-2]

        stack = []
        stack.append(convex_set[0])
        stack.append(convex_set[1])
        i = 2
        while (i < convex_set.shape[0]):

            if (len(stack)-2 < 0):  # top -> next may be undefined, this part is not in the original text
                stack.append(convex_set[i])
                i = i + 1
                continue

            p1 = stack[len(stack)-2]  # top -> next
            p2 = stack[len(stack)-1]  # top

            if (self.Left(p1, p2, convex_set[i]) > 0):
                stack.append(convex_set[i])
                i = i + 1
            else:
                stack.pop()

        convex_hull = np.array(stack)
        return convex_hull

    
    # returns the convex hull points in counterclockwise order
    def run(self):

        # Graham's Scan procedure
        self.find_lowest()
        self.quickSort(self.points[1:self.points.shape[0]], 0, self.points.shape[0]-2)
        convex_set = self.squash()
        convex_hull = self.graham_scan(convex_set)
        
        print("Graham's Scan CH Points:")
        print(convex_hull)
        return convex_hull
