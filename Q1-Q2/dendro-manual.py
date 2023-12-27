import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

l = np.array([[0.,1.,1.,2.],
[4.,5.,1.,2.],
[6.,7.,1.,2.],
[8.,3.,1.41,3.],
[9.,10.,1.41,4.],
[11.,2.,2.,4.],
[12.,13.,4.24,8.]])
dendrogram(l,labels='p1 p2 p3 p4 p5 p6 p7 p8'.split())
plt.show()