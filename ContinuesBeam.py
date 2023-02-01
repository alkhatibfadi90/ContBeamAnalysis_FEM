import numpy as np
from math import *
import matplotlib.pyplot as plt




class contBeam:
    def __init__(self, bw, h, E, nodes, elements):
        self.bw = bw
        self.h = h
        self.A = self.bw * self.h
        self.I = self.bw * (self.h ** 3) / 12
        self.E = E
        self.Node = np.array(nodes).astype(float)
        self.Elem = np.array(elements).astype(int)






        self.DoF = 2
        self.nEDoF = 2 * self.DoF
        self.nE = len(self.Elem)
        self.nN = len(self.Node)
        self.nNDoF = self.nN * self.DoF
        self.EDoF = np.arange(0, self.nNDoF)
        self.EDoF = np.reshape(self.EDoF, (-1, self.DoF))

        self.wLoad = np.array([self.nE, 2])
        self.pLoad = np.zeros_like(self.Node)

        self.Support = np.ones_like(nodes).astype(int)

        self.Force = np.zeros([self.nE, self.nEDoF])
        self.Displacement = np.zeros([self.nE, self.nEDoF])

        dist = self.Node[self.Elem[:, 1], :] - self.Node[self.Elem[:, 0], :]
        self.L = np.sqrt(dist ** 2).sum(axis=1)

        # print(self.EDoF)
    def BeamMatrix(self, l):
        Matrix = np.array([[12, 6 * l, -12, 6 * l],
                            [6 * l, 4 * l ** 2, -6 * l, 2 * l ** 2],
                            [-12, -6 * l, 12, -6 * l],
                            [6 * l, 2 * l ** 2, -6 * l, 4 * l ** 2]])


        ElemMatrix = self.E * self.I * Matrix / l**3
        return ElemMatrix

    def Loading(self, pl, wl):
        self.EqvLoad = np.zeros([self.nE, self.nEDoF])



        #Point Loading
        if pl:
            for p in pl:
                e = p[0]
                l = self.L[e]
                a = p[1]
                b = l - a
                P = p[2]

                self.EqvLoad[e, 0] += P * b**2 * (l + 2 * a) / l**3
                self.EqvLoad[e, 1] += P * a * b**2 / l**2
                self.EqvLoad[e, 2] += P * a**2 * (l + 2 * b) / l**3
                self.EqvLoad[e, 3] += -P * b * a**2 / l**2


        #Distributed Loading
        if wl:
            for w in wl:
                e = w[0]
                l = self.L[e]
                wi = w[1]
                wj = w[2]
                self.EqvLoad[e, 0] += l * (21 * wi + 9 * wj) / 60
                self.EqvLoad[e, 1] += l * (l * (3 * wi + 2 * wj)) / 60
                self.EqvLoad[e, 2] += l * (9 * wi + 21 * wj) / 60
                self.EqvLoad[e, 3] += l * (l * (-2 * wi - 3 * wj)) / 60


        for i in range(self.nE):
            self.pLoad[self.Elem[i, 0], 0] += self.EqvLoad[i, 0]
            self.pLoad[self.Elem[i, 0], 1] += self.EqvLoad[i, 1]
            self.pLoad[self.Elem[i, 1], 0] += self.EqvLoad[i, 2]
            self.pLoad[self.Elem[i, 1], 1] += self.EqvLoad[i, 3]

        return self.EqvLoad, self.pLoad

    def Supporting(self, support):

        for ss in support:

            self.Support[ss[0], : ] = ss[1], ss[2]

        return  self.Support





    def analysis (self):

        ElemK = np.zeros([self.nE, self.nEDoF, self.nEDoF])
        K = np.zeros([self.nNDoF, self.nNDoF])


        for i in range(self.nE):
            ElemK[i] = self.BeamMatrix(self.L[i])
            ElemDoF = self.EDoF[self.Elem[i]].flatten()
            K[np.ix_(ElemDoF, ElemDoF)] += ElemK[i]

        freeDoF = self.Support.flatten().nonzero()[0]

        kff = K[np.ix_(freeDoF, freeDoF)]
        p = self.pLoad.flatten()
        pf = p[freeDoF]
        uf = np.linalg.solve(kff, pf)
        u = self.Support.astype(float).flatten()
        u[freeDoF] = uf
        u = u.reshape(self.nN, self.DoF)
        # print(u)
        uElem = np.concatenate((u[self.Elem[:, 0]], u[self.Elem[:, 1]]), axis=1)



        for i in range(self.nE):
            self.Force[i] = np.dot(ElemK[i], uElem[i]) - self.EqvLoad[i]
            self.Displacement[i] = uElem[i]


        return self.Force, self.Displacement


    def plot(self, scale):
        fig, ax = plt.subplots(3)
        #loading
        for i in range(self.nE):
            xi, xj = self.Node[self.Elem[i, 0], 0], self.Node[self.Elem[i, 1], 0]
            yi, yj = self.Node[self.Elem[i, 0], 1], self.Node[self.Elem[i, 1], 1]
            ax[0].plot((xi, xj), (yi, yj), 'b', linewidth= 1)

        for i in range(self.nE):
            dxi, dxj = self.Node[self.Elem[i, 0], 0], self.Node[self.Elem[i, 1], 0]
            dyi = self.Node[self.Elem[i, 0], 1] + self.Displacement[i, 0] * scale
            dyj = self.Node[self.Elem[i, 1], 1] + self.Displacement[i, 2] * scale
            ax[0].plot([dxi, dxj], [dyi, dyj], 'r', linewidth=2)
            ax[0].text(dxi, dyi, str(round(dyj*1000, 0))+'mm', rotation=90)

        #Moment
        ax[1].invert_yaxis()
        for i in range(self.nE):
            mxi, mxj = self.Node[self.Elem[i, 0], 0], self.Node[self.Elem[i, 1], 0]
            myi, myj = self.Node[self.Elem[i, 0], 1], self.Node[self.Elem[i, 1], 1]
            ax[1].plot((mxi, mxj), (myi, myj), 'b', linewidth= 1)

        for i in range(self.nE):
            mrxi, mrxj = self.Node[self.Elem[i, 0], 0], self.Node[self.Elem[i, 1], 0]
            mryi = - self.Force[i, 1]
            mryj = self.Force[i, 3]
            ax[1].plot([mrxi, mrxi, mrxj, mrxj], [0, mryi, mryj, 0], 'r', linewidth=2)
            ax[1].fill([mrxi, mrxi, mrxj, mrxj], [0, mryi, mryj, 0], 'c', alpha=0.3)
            ax[1].text(mrxi, mryi, str(round(mryj, 2))+'kN.m', rotation=90)

        #shear
        ax[2].invert_yaxis()
        for i in range(self.nE):
            mxi, mxj = self.Node[self.Elem[i, 0], 0], self.Node[self.Elem[i, 1], 0]
            myi, myj = self.Node[self.Elem[i, 0], 1], self.Node[self.Elem[i, 1], 1]
            ax[2].plot((mxi, mxj), (myi, myj), 'b', linewidth= 1)

        for i in range(self.nE):
            mrxi, mrxj = self.Node[self.Elem[i, 0], 0], self.Node[self.Elem[i, 1], 0]
            mryi = - self.Force[i, 0]
            mryj = self.Force[i, 2]
            ax[2].plot([mrxi, mrxi, mrxj, mrxj], [0, mryi, mryj, 0], 'r', linewidth=2)
            ax[2].fill([mrxi, mrxi, mrxj, mrxj], [0, mryi, mryj, 0], 'orange', alpha=0.3)
            ax[2].text(mrxi, mryi, str(round(mryj, 2))+'kN', fontsize= 10, rotation=90)



        fig.tight_layout(pad=0.5)
        plt.show()



        # for i in range(self.nE):
        #        print(LoadEq.shape)
        #
        # print(K)



# nodes = [[0,0],[2.5, 0], [5,0], [7.5,0], [10,0],[12.5,0], [15,0]]
# elements = [[0,1], [1,2], [2,3],[3,4], [4,5],[5,6]]

# nodes = [[0,0], [5,0], [10,0]]
# elements = [[0,1], [1,2]]
# pload = [[0, 5, -10]]
# wload = [[0, -10, -10],[1,-10,-10]]
# support = [[0,0,1], [2,0,1]]
bw = 0.5
h = 1.0
E = 30100
nodes = []
elements = []
for i in range(20):
    n = [i, 0]
    nodes.append(n)

for i in range(len(nodes) - 1):
    e = [i, i + 1]
    elements.append(e)

pload = [[4, 0.9, -10]] #elem pos2 value
wload = [[14, -10, -10], [15, -10, -10], [16, -10, -10], [17, -10, -10], [18, -10, -10]] #elem valuestart valueend
support = [[0, 0, 1], [9, 0, 1], [19,0,0]]

beam = contBeam(bw, h, E, nodes, elements)
load = beam.Loading(pl=pload, wl=wload)
supp = beam.Supporting(support)
allload = beam.pLoad

analy = beam.analysis()
# print(analy)
beam.plot(1)

