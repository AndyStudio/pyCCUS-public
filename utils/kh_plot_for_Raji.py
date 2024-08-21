import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import xlrd
import xlwt
import os


threshold=0.005

## Please load your data here with the correct path
perm = np.load('perm.npy')
z_coord = np.load('z_coord.npy')


# Define a function to filter out the low permeability layers
def avg_smooth(x,shift,threshold):
    n = len(x)
    l = n - shift + 1
    rst = np.zeros(l)
    thre_mask = np.zeros(l)
    for i in range(l):
        selected_prop = x[i:i+shift]
        x_avg = sum(selected_prop)/len(selected_prop)
        rst[i] = x_avg
        if x_avg >= threshold:
            thre_mask[i] = 1
    return rst,thre_mask

# Define a function to compute grid block size
def calc_grid_length(x):
    new_x = x[x!=-99]
    if len(new_x)>=3:
        elm1 = new_x[1]-new_x[0]
        elm2 = new_x[-1]-new_x[-2]
        dist = np.diff(new_x)
        mid = 0.5*(dist[:-1]+dist[1:])
        new_x_length = np.hstack((elm1, mid, elm2))
        rst = np.zeros(x.shape)
        rst[x!=-99]=new_x_length
        return rst
    if len(new_x) == 2:
        elm = new_x[1]-new_x[0]
        new_x_length = np.array([elm,elm])
        rst = np.zeros(x.shape)
        rst[x!=-99]=new_x_length
        return rst
    else:
        return np.zeros(x.shape)

# Plot your k*h map here
m,n = perm.shape[0],perm.shape[1]
kh_map = np.zeros((m,n))
for ii in range(m):
    for jj in range(n):
        _, mask = avg_smooth(perm[ii,jj,:],1,threshold)
        k_perm = perm[ii,jj,:]
        h_perm = np.abs(calc_grid_length(z_coord[ii,jj,:]))
        kh_map[ii,jj] = sum(k_perm*h_perm*mask)

fig = px.imshow(np.rot90(kh_map))
fig.show()
