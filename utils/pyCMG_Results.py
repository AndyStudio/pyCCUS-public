'''
Author: Yunan Li
Date: 2023/02/17
Goal: This file contains class and functions to interact with CMG Results in Python
Class:  1. pycmgresults: read the simulation rst from CMG in Python format
        2. pycmgpostcalc: calculations for CCS related metrics.
'''

import os
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt   
# import plotly.express as px


class pycmgresults():
    def __init__(self):
        super().__init__()

    def read_gslib2csv(self, path2file):
        """
        This function reads the absolute path to file, and return a dataframe df
        The file should be gslib file exported from Petrel
        """
        with open(path2file) as file:
            lines = file.readlines()

        num_col = int(lines[1].split()[0])
        name_col = [x.split()[0] for x in lines[2:2+num_col]]

        list_all = []
        for i in range(num_col+2, len(lines)):
        #     val_list = [float(x) for x in lines[i].split()]
            val_list = lines[i].split()
            list_all.append(val_list)
        arr_all = np.array(list_all, dtype=float)
        df = pd.DataFrame(data=arr_all, columns=name_col)
        return df

    def gslib2arr(self,df,query):
        """
        This function converts gslib in df to array
        """
        i_idx = sorted(df['i_index'].unique())
        j_idx = sorted(df['j_index'].unique())
        k_idx = sorted(df['k_index'].unique())
        
        prop_arr = np.zeros((len(i_idx),len(j_idx),len(k_idx)))
        for arr_k,kk in enumerate(k_idx):
            df_kk = df[df['k_index'] == kk]
            for arr_j, jj in enumerate(j_idx):
                prop_arr[:,arr_j,arr_k] = np.array(df_kk[df_kk['j_index'] == jj][query])
                
        return i_idx,j_idx,k_idx,prop_arr
    

    def read_rwo2csv(self, 
                     path2file, 
                     save2csv=None):
        """
        Goal: reads rwo files to csv format. The rwo file is prop values at X, Y location.

        Input:  path2file directions the function to the rwo file.
                save2csv: path+name of the csv file to be saved, if not None.
        Output: csv file includes colums: X, Y, prop @ time steps output in rwo files. 
                (Num of rows should be the number of grids in a layer of the model)
                example format of a column: "Gas Saturation_2023-Jan-01"
        """
        # read data from file
        with open(path2file) as file:
            lines = file.readlines()
        # organize data in dict format    
        cache = {}
        count = 0
        
        for i in range(len(lines)):
            try:
                # This means a new time step
                if lines[i].split()[1] == 'TIME:':
                    if count > 0:
                        cache[key] = pressure
                    count += 1
                    pressure = []
    #                 key = lines[i+1].split(':')[-1]+'_'+str(count)
    #                 key = lines[i+1].split(':')[-1]+'_'+lines[i].split()[-1]
                    prop_name = ' '.join(lines[i+1].split(':')[-1].split())
                    key = prop_name+'_'+lines[i].split()[-1]

                if lines[i].split()[0] == '**' or lines[i].split()[0] == '<' or not lines[i].split():
                    pass
                else:
                    x = float(lines[i].split()[0])
                    y = float(lines[i].split()[1])
                    prop = float(lines[i].split()[2])
                    pressure.append((x,y,prop))
            except:
                pass
        # Record the last item
        cache[key] = pressure
            
        # convert cache in dict to a pd dataframe
        df = pd.DataFrame({})
        df['X'] = np.array(cache[list(cache.keys())[0]])[:,0]
        df['Y'] = np.array(cache[list(cache.keys())[0]])[:,1]
        for item in cache.keys():
            df[item] = np.array(cache[item])[:,2]
        
        # If need to save the data in CSV format
        if save2csv is not None:
            df.to_csv(save2csv, index=False)
        return df
    
    def rwodf2arr(self, df, query, x_dir_key='X', y_dir_key='Y'):
        """
        Input:  df - output from read_rwo2csv function above in df format
                query - query time step to get a 2d array for a certain layer at a certain time step. (example="Gas Saturation_2027-Jan-01")

        Output: np array (n,m). X increasing per col num, and Y decreasing per row num

                ^
                |
            Y   |   prop_val
           (n)  |
                0   --  --  -- > 
                    X (num=m)
        """
        xx = np.sort(df[x_dir_key].unique())
        yy = np.sort(df[y_dir_key].unique())
        m,n = len(xx), len(yy)
        pres_arr = np.zeros((n,m))
        pres = df
        selected_time = query

        for i,x in enumerate(xx):
            new_pres = pres[pres[x_dir_key] == x].sort_values(by=[y_dir_key],ascending=False)
            pres_arr[:len(new_pres[selected_time]),i] = new_pres[selected_time].to_numpy()


        return pres_arr
    
    def rwodf2arr_ReefRidge(self,x):
        m,n = x.shape
        if n != 97:
            print('Check array shape')
        a = np.zeros((74,97))
        a[-m:,:] = x
        return a
    

    def rwo_reader2df(self, folder, sim, prop, layer_nums):
        """
        Inputs: 1. folder: path to the sim run folder, example="../data/EPA_baseline/"
                2. sim: simulation run name, example="EPA_baseline_dev2East_debug"
                3. prop: CMG keyword for property 'SG', 'PRES', 'VERDSPLGEO', etc.
                4. layer_nums: list or array in 1d
        """

        for ll in layer_nums:
            globals()[f'self.df{prop}_lyr{ll}'] = self.read_rwo2csv(f'{folder}{sim}_{prop}_layer{ll}.rwo')

        return print(f'Reading {sim} property of {prop} is done in global var .....')
    

    def rwo_reader2arr(self, folder, sim, prop, layer_nums, time_query, x_dir_key='X', y_dir_key='Y'):
        """
        Goal: combine rwo_reader2df and rwodf2arr in a 4d (n,m,z,t) array for output.

        Inputs: 1. folder: path to the sim run folder, example="../data/EPA_baseline/"
                2. sim: simulation run name, example="EPA_baseline_dev2East_debug"
                3. prop: CMG keyword for property 'SG', 'PRES', 'VERDSPLGEO', etc.
                4. layer_nums: list or array in 1d
                5. query: in format of [list] or arr for years. example=['Gas Saturation_2027-Jan-01', ..., 'Gas Saturation_2100-Jan-01']
        """
        df0 = self.read_rwo2csv(os.path.join(folder, f'{sim}_{prop}_layer{layer_nums[0]}.rwo'))
        arr0 = self.rwodf2arr(df=df0, query=time_query[0], x_dir_key='X', y_dir_key='Y')
        n,m = arr0.shape
        z = len(layer_nums)
        t = len(time_query)

        rst_arr = np.zeros((n,m,z,t))
        for il,ll in enumerate(layer_nums):
            globals()[f'df{prop}_lyr{ll}'] = self.read_rwo2csv(os.path.join(folder, f'{sim}_{prop}_layer{ll}.rwo'))
            for it, tt in enumerate(time_query):
                globals()[f'arr{prop}_lyr{ll}_{tt}'] = self.rwodf2arr(df=globals()[f'df{prop}_lyr{ll}'], query=tt, x_dir_key=x_dir_key, y_dir_key=y_dir_key)
                rst_arr[:,:,il,it] = globals()[f'arr{prop}_lyr{ll}_{tt}']
        return rst_arr
            

        


class pycmgpostcalc():
    def __init__(self):
        super().__init__()


    def calc_CO2plume(self, fourDarr, time_query, co2threshold, prop, layer_nums, sizemul=None):
        """
        This only includes the maximum land co2 plume method.
        Inputs:
        Outputs:
        """
        
        # combo_SG = pycmgresults.rwodf2arr(globals()[f'df{prop}_lyr{layer_nums[0]}'], query=time_query)
        # m,n = combo_SG.shape
        # for ll in layer_nums:
        #     globals()[f'arr{prop}_lyr{ll}'] = pycmgresults.rwodf2arr(globals()[f'df{prop}_lyr{ll}'], query=time_query)
        #     combo_SG = np.stack((combo_SG,globals()[f'arr{prop}_lyr{ll}']), axis=2)

        # max_combo_SG = np.max(combo_SG, axis=2)

        n,m,z,t = fourDarr.shape
        max_combo_SG = np.max(fourDarr, axis=2)
        plume = max_combo_SG > co2threshold
        if sizemul is None:
            plumesize = np.sum(plume,axis=(0,1))/m/n
            print('Plume size is ratio of num of grids with CO2 to total num of grids, wtihout a sizemul defined ...')
        else:
            plumesize = np.sum(plume,axis=(0,1))/m/n*sizemul

        all_points, all_outlines = [], []
        for tt in range(t):
            pts, outline = self.plume_outline(plume[:,:,tt])
            all_points.append(pts)
            all_outlines.append(outline)

        # _, globals()[f'co2outline_{time_query}'] = self.plume_outline(plume)

        return plumesize,all_points,all_outlines

    def plume_outline(self, x, flag=True):
        m,n = x.shape
        rst_list = []
        plt_list1, plt_list2 = [],[]
        for i in range(m):
            idx = np.where(x[i,:]==flag)[0]
            if len(idx) == 0:
                pass
            else:
                idx_max = np.max(idx)
                idx_min = np.min(idx)
                if idx_max == idx_min:
                    rst_list.append((i,idx_max))
                    plt_list2.append((i,idx_max))
                else:
                    rst_list.append((i,idx_min))
                    rst_list.append((i,idx_max))
                    plt_list1.append((i,idx_min))
                    plt_list2.append((i,idx_max))
                    
        if len(rst_list)>=3:
            plt_list2.append(plt_list1[-1])
            plt_list2 = [plt_list1[0]] + plt_list2

        else:
            print('Please check the plume shape because it is unlikely to draw a boundary .....')
            
        entire_points = rst_list
        up_half_points = plt_list1
        bottom_half_points = plt_list2
        return entire_points, [up_half_points,bottom_half_points]
    
    # def calc_CO2plume_vs_time(self, co2threshold, prop, layer_nums, time_steps):
    #     co2plumesize = []
    #     for tt in time_steps:
    #         time_query = f'Gas Saturation_{tt}-Jan-01'
    #         plumesize = self.calc_CO2plume(time_query, co2threshold, prop, layer_nums, sizemul=1)
    #         co2plumesize.append(plumesize)

    #     return co2plumesize
    
    def plt_CO2plume(self, time_steps, savefig=False):
        
        for tt in time_steps:
            time_query = f'Gas Saturation_{tt}-Jan-01'
            plt.plot(np.array(globals()[f'co2outline_{time_query}'][0])[:,1],np.array(globals()[f'co2outline_{time_query}'][0])[:,0],'r')
            plt.plot(np.array(globals()[f'co2outline_{time_query}'][1])[:,1],np.array(globals()[f'co2outline_{time_query}'][1])[:,0],'r',label=f't={tt}')
       
        plt.xlabel('I index', fontsize=14)
        plt.ylabel('J index', fontsize=14)
        plt.xlim([0,33])
        plt.ylim([0,33])
        plt.legend()
        plt.legend(prop={'size': 10})
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.grid()
        if savefig == True:
            plt.savefig("CO2plumes.png",dpi=300)
        plt.show()


    def ijk2xyz(self,i,X):
        return X[i]

    def calc_dist(self,A,B):
        x1,y1 = A[0],A[1]
        x2,y2 = B[0],B[1]
        d = np.sqrt((x1-x2)**2+(y1-y2)**2)
        return d 
    
    def calc_plume_edge_dist2inj(self,pts, inj, X, Y):
        dist = []
        norm_dist_params = []
        bound_dist = []
        t = len(pts)
        xy_inj = (self.ijk2xyz(inj[0],X), self.ijk2xyz(inj[1],Y))
        for tt in range(t):
            dd = []
            for pp in pts[tt]:
                xy_pt = (self.ijk2xyz(pp[0],X), self.ijk2xyz(pp[1],Y))
                d = self.calc_dist(A=xy_inj,B=xy_pt)
                dd.append(d)
            if len(dd) > 0:
                dist.append(dd)
                mu, std = norm.fit(dd)
                norm_dist_params.append((mu,std))
                bound_dist.append((min(dd),max(dd)))
            else:
                # Case when t=0
                dist.append(0)
                norm_dist_params.append((0,0))
                bound_dist.append((0,0))
        return dist, norm_dist_params, bound_dist
    


    def avg_smooth(self,x,shift,threshold):
        """
        This function calculates smooth curve by average.
        Input x: 1d array or list
        Input shift: int. controls number of elements next to i should be considered for average smooth.
        Input threshould: int or float, just a scalar. Compare the averaged value to threshold 
        Output rst: smoothed 1d array in len of len(x)-shift+1. 
        Output thre_mask: 1d array tell which element >= threshold (value=1) and which is not (value=0)
        """
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

    def calc_nan_mask(self,x,nan_val):
        """
        Input x: must be an array (could be 1d, 2d, 3d, etc.)
        Input nan_val: the nan val we would like to identify
        Output mask: array of the same shape as x. (val=1 means valid and val=0 means found nan value)
        """
        mask = np.ones(x.shape)
        mask[x==nan_val]=0
        return mask

    def calc_grid_length(self,x,nan_val=None):
        """
        Input x: 1d array represents grid location in unit meter or ft
        Output: grid block length of the side.
        Note: approximation at the edge of the array.
        Note: due to the fact we calc grid length, we set it to 0 for nan values.
        """
        if nan_val is None:
            if len(x)>=3:
                elm1 = x[1]-x[0]
                elm2 = x[-1]-x[-2]
                dist = np.diff(x)
                mid = 0.5*(dist[:-1]+dist[1:])
                return np.hstack((elm1, mid, elm2))
            elif len(x) == 2:
                return np.array([x[1]-x[0],x[1]-x[0]])
            else:
                raise TypeError("Input 1d array must include at least 2 elements")
        else:
            new_x = x[x!=nan_val]
            if len(new_x)>=3:
                elm1 = new_x[1]-new_x[0]
                elm2 = new_x[-1]-new_x[-2]
                dist = np.diff(new_x)
                mid = 0.5*(dist[:-1]+dist[1:])
                new_x_length = np.hstack((elm1, mid, elm2))
                rst = np.zeros(x.shape)
                rst[x!=nan_val]=new_x_length
                return rst
            if len(new_x) == 2:
                elm = new_x[1]-new_x[0]
                new_x_length = np.array([elm,elm])
                rst = np.zeros(x.shape)
                rst[x!=nan_val]=new_x_length
                return rst
            else:
                return np.zeros(x.shape)


    def calc_combo_map(self,x,y,w):
        """
        This function calculates combination of 2 maps.
        Formula: combo_map = w*map1+(1-w)*map2
        Note: map1 and map2 are normalized to combine at the same scale

        Input x: 1d or 2d array
        Input y: 1d or 2d array
        Input w: scalar in range of 0 and 1. 
        Output combo_xy: calculated combo map based on 2 input maps with the same shape.
        """
        if x.shape == y.shape:
            x_min,x_max = x.min(),x.max()
            y_min,y_max = y.min(),y.max()
            norm_x = (x-x_min)*(1/(x_max-x_min))
            norm_y = (y-y_min)*(1/(y_max-y_min))
            combo_xy = w*norm_x+(1-w)*norm_y
            return combo_xy
        else:
            raise TypeError("Input arrays must be of the same shape")

    def calc_kh_map(self, z_coord, prop, shift, threshold):
        """
        This function calculates kh map based on 3d arrays exported from Petrel or in CMG.
        Formula: 

        Input z_coord: 3d array shows z direction in meters or ft shape->[m,n,k]
        Input prop: 3d array represents a property selected. (could be permeability, porosity, etc.) shape->[m,n,k]
        Input shift: int scalar. controls number of elements next to i should be considered for average smooth.
        Input threshould: int or float, just a scalar. Compare the averaged value to threshold 
        Output khmap: calculated kh map in 2d array shape. shape->[m,n]
        """
        # Updated this function so we keep shift = 1
        shift = 1
        ############################################
        m,n = prop.shape[0],prop.shape[1]
        kh_map = np.zeros((m,n))
        for ii in range(m):
            for jj in range(n):
                _, mask = self.avg_smooth(prop[ii,jj,:],shift,threshold)
                k_perm = prop[ii,jj,:]
                h_perm = np.abs(self.calc_grid_length(x=z_coord[ii,jj,:],nan_val=-99))
                # h_perm = -np.diff(z_coord[ii,jj,:])
                kh_map[ii,jj] = sum(k_perm*h_perm*mask)
        return kh_map

    def calc_Vgrid(self,x_coord,y_coord,z_coord):
        """
        This function calculates the grid block volume given x,y,z coordinate from Petrel
        Input x_coord: shape->[m,n,k]
        Input y_coord: shape->[m,n,k]
        Input z_coord: shape->[m,n,k]
        All inputs show have exactly the same shape. m --> i, n --> j, k --> k
        An example size for all inputs is (74, 97, 711) for the largest Petrel model from SPR.
        Output: grid block cell volume
        """
        length_i, length_j, length_k = np.zeros(x_coord.shape),np.zeros(y_coord.shape),np.zeros(z_coord.shape)
        i,j,k = x_coord.shape
        for jj in range(j):
            for kk in range(k):
                length_i[:,jj,kk] = self.calc_grid_length(x_coord[:,jj,kk],-99)

        for ii in range(i):
            for kk in range(k):
                length_j[ii,:,kk] = self.calc_grid_length(y_coord[ii,:,kk],-99)

        for ii in range(i):
            for jj in range(j):
                length_k[ii,jj,:] = self.calc_grid_length(z_coord[ii,jj,:],-99)
                
        V_grid = np.abs(length_i*length_j*length_k)
        xnan_mask = self.calc_nan_mask(x_coord,-99)
        ynan_mask = self.calc_nan_mask(y_coord,-99)
        znan_mask = self.calc_nan_mask(z_coord,-99)
        nan_mask = xnan_mask*ynan_mask*znan_mask
        
        return V_grid*nan_mask
    

    
    def outline_idx_from_2D_TrueFalse_array(self, x, flag=True):
        """
        x is a 2D matrix with True and False. 
        False means not to ignore. 
        True means the area we need to find the index of the boundary.
        Limitation: only 1 outline for entire set. If there are multiple outlines expected, need to improve.
        """
        m, _ = x.shape
        collect_bnds_idx = []
        for i in range(m):
            idx = np.where(x[i,:]==flag)[0]
            if len(idx) == 0:
                pass
            else:
                idx_max = np.max(idx)
                idx_min = np.min(idx)
                idx_y = m-1-i
                if idx_max == idx_min:
                    collect_bnds_idx.append((idx_max, idx_y))
                else:
                    collect_bnds_idx = [(idx_min, idx_y)] + collect_bnds_idx
                    collect_bnds_idx = collect_bnds_idx + [(idx_max, idx_y)]

        return collect_bnds_idx

    def ijindex_to_XYcoord(self, shape_idx, matX, matY):
        """
        shape_idx: list of tuples [(i1,ji),...,(in,jn)]. Index of boundary points.
        matX: 2D matrix array for X coordinates in ft or lat for all grids
        matY: 2D matrix array for Y coordinates in ft or long for all grids
        """
        plot_pts = []
        for ii, idx in enumerate(shape_idx):

            pt1x = matX[idx]
            pt1y = matY[idx]

            pt2x = matX[shape_idx[ii-1]]
            pt2y = matY[shape_idx[ii-1]]

            plot_pts.append(([pt1x, pt2x],[pt1y, pt2y]))
        return plot_pts