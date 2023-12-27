from cmath import inf
import numpy as np
import pandas as pd

# to get each element in a cluster to compare ditances between each combination of clusters
def flattenCluster(c):
    flat = []
    if type(c) != tuple:
        c = (c,)
    for x in c:
        if type(x) == tuple:
            flat.extend(flattenCluster(x))
        else:
            flat.append(x)
    return tuple(flat)

# calculate minimum ditances between clusters
def clusterD(c1,c2,d):
    if type(c1) != tuple:
        c1 = (c1,)
    if type(c2) != tuple:
        c2 = (c2,)

    c1 = flattenCluster(c1)
    c2 = flattenCluster(c2)

# we start with distance = infinity and try to minimize it
    distance = inf
    for i in c1:
        for j in c2:
            if d.loc[i,j]<distance:
                distance = d.loc[i,j]
    return distance

# dissimilarity matrix
d = pd.DataFrame([[0,1,4],[1,0,2],[4,2,0]])
index = []
# giving points names
for x in range(1,4):
    index.append(f'p{x}')
d.index = index
d.columns = index

# setting diagonal with infinity so we can get minimum distance that is not between a point with itself
d[d==0]=inf
# saving every iteration and counting them
iterations = [d]
it=0
while len(iterations[-1])>1 and it < 10:
    d2 = iterations[-1]
    mini = np.min(d2.values)
    # indexes where the minimum distance is
    i = np.where(d2 == mini)
    clustered = []
    new_clusters = []
    # merge clusters that has minimum distances if they have not been merged yet
    for x in range(len(i[0])):
        if i[0][x]>i[1][x] and i[0][x] not in clustered and i[1][x] not in clustered:
            new_clusters.append((d2.index[i[1][x]],d2.index[i[0][x]]))
            clustered.append(i[0][x])
            clustered.append(i[1][x])

    # add the unmerged clusters
    for x,_ in enumerate(d2.index):
        if x not in clustered:
            new_clusters.append(d2.index[x])
            clustered.append(x)
    iterations.append(
        pd.DataFrame(
            np.zeros(
                (len(new_clusters),len(new_clusters))
                )
            )
        )
    iterations[-1].index = new_clusters
    iterations[-1].columns = new_clusters

    # calculate distances after mergin
    for i,ii in enumerate(iterations[-1].index):
        for j,jj in enumerate(iterations[-1].index):
            if i == j:
                iterations[-1].iloc[i,j] = inf
            else:
                iterations[-1].iloc[i,j] = clusterD(ii,jj,d)
    print('-'*13,f"Iteration: {it+1}",'-'*13)
    print(iterations[-1])
    it+=1

print('-'*50)
c = []
print("Clusters: ",end='')
for x in iterations[-1].index:
    c.append(x)
print(c)