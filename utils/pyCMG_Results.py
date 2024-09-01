'''
Author: Yunan Li
Date: 2023/02/17
Goal: This file contains class and functions to interact with CMG Results in Python
Class:  1. pycmgresults: read the simulation rst from CMG in Python format
        2. pycmgpostcalc: calculations for CCS related metrics.
'''

import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import norm

import matplotlib.pyplot as plt   
# import plotly.express as px
from tqdm import tqdm
import pickle
from scipy import interpolate
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.interpolate import griddata




class pycmgresults():
    def __init__(self):
        super().__init__()
        self.XY2arr_interp_method = "cubic"  # options = {‘linear’, ‘nearest’, ‘cubic’}
        self.XY2arr_interp_num_x = 100
        self.XY2arr_interp_num_y = 100

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
    

    
    def xy_interp_to_arr(self, df, num_x, num_y, interp_method, query, x_dir_key='X', y_dir_key='Y'):
        """
        interp_method = {‘linear’, ‘nearest’, ‘cubic’}
        num_x, num_y = int. Number of query pts along x or y axis.
        x, y, values = df columns
        """

        x = df[x_dir_key].values
        y = df[y_dir_key].values
        val = df[query].values

        # Define grid
        xi = np.linspace(np.min(x), np.max(x), num_x)
        yi = np.linspace(np.min(y), np.max(y), num_y)
        x_new, y_new = np.meshgrid(xi, yi)

        val_new = griddata((x, y), val, (x_new, y_new), method=interp_method)

        return x_new, y_new, val_new
    
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
    

    def rwo_reader2arr(self, folder, sim, prop, layer_nums, time_query, x_dir_key, y_dir_key):
        """
        Goal: combine rwo_reader2df and rwodf2arr in a 4d (n,m,z,t) array for output.

        Inputs: 1. folder: path to the sim run folder, example="../data/EPA_baseline/"
                2. sim: simulation run name, example="EPA_baseline_dev2East_debug"
                3. prop: CMG keyword for property 'SG', 'PRES', 'VERDSPLGEO', etc.
                4. layer_nums: list or array in 1d
                5. query: in format of [list] or arr for years. example=['Gas Saturation_2027-Jan-01', ..., 'Gas Saturation_2100-Jan-01']
        """
        df0 = self.read_rwo2csv(os.path.join(folder, f'{sim}_{prop}_layer{layer_nums[0]}.rwo'))

        x_new, y_new, arr0 = self.xy_interp_to_arr(df=df0, 
                                                   num_x=self.XY2arr_interp_num_x, num_y=self.XY2arr_interp_num_y, interp_method=self.XY2arr_interp_method, 
                                                   query=time_query[0], x_dir_key=x_dir_key, y_dir_key=x_dir_key)

        # Retire the old version, b/c it does not handle the case that X and Y do not align
        # arr0 = self.rwodf2arr(df=df0, query=time_query[0], x_dir_key='X', y_dir_key='Y')

        n,m = arr0.shape
        z = len(layer_nums)
        t = len(time_query)

        rst_arr = np.zeros((n,m,z,t))
        for il,ll in enumerate(layer_nums):
            globals()[f'df{prop}_lyr{ll}'] = self.read_rwo2csv(os.path.join(folder, f'{sim}_{prop}_layer{ll}.rwo'))
            for it, tt in enumerate(time_query):
                _, _, globals()[f'arr{prop}_lyr{ll}_{tt}'] = self.xy_interp_to_arr(df=globals()[f'df{prop}_lyr{ll}'], 
                                                                                   num_x=self.XY2arr_interp_num_x, 
                                                                                   num_y=self.XY2arr_interp_num_y, 
                                                                                   interp_method=self.XY2arr_interp_method, 
                                                                                   query=tt, x_dir_key=x_dir_key, y_dir_key=y_dir_key)
                # Retire the old version, b/c it does not handle the case that X and Y do not align
                # globals()[f'arr{prop}_lyr{ll}_{tt}'] = self.rwodf2arr(df=globals()[f'df{prop}_lyr{ll}'], query=tt, x_dir_key=x_dir_key, y_dir_key=y_dir_key)
                rst_arr[:,:,il,it] = globals()[f'arr{prop}_lyr{ll}_{tt}']
        return x_new, y_new, rst_arr
            

        


class pycmgpostcalc():
    def __init__(self):
        super().__init__()
        self.rescale_method = 'linear'
        self.rescaleX_num_grids = 200
        self.rescaleY_num_grids = 200


    def load_Etchegoin_6x6_data(self):
        if sys.platform == 'win32':
            self.CMG_coord_folder = 'E:\\CUSP_win\\CCUS\\data\\6x6model\\'
            self.Ecgn_x = np.load(os.path.join(self.CMG_coord_folder, 'Etchegoin_x_coord.npy'))
            self.Ecgn_y = np.load(os.path.join(self.CMG_coord_folder, 'Etchegoin_y_coord.npy'))
            self.Ecgn_z = np.load(os.path.join(self.CMG_coord_folder, 'Etchegoin_z_coord.npy'))
        elif sys.platform == 'darwin':
            self.CMG_coord_folder = '/Users/yunanli/Library/CloudStorage/OneDrive-Stanford/1.CUSP/CCUS/data/6x6model'
            self.Ecgn_x = np.load(os.path.join(self.CMG_coord_folder, 'Etchegoin_x_coord.npy'))
            self.Ecgn_y = np.load(os.path.join(self.CMG_coord_folder, 'Etchegoin_y_coord.npy'))
            self.Ecgn_z = np.load(os.path.join(self.CMG_coord_folder, 'Etchegoin_z_coord.npy'))
        elif sys.platform == 'linux':
            self.CMG_coord_folder = '/scratch/users/yunanli/pyCCUS/data/6x6model'
            self.Ecgn_x = np.load(os.path.join(self.CMG_coord_folder, 'Etchegoin_x_coord.npy'))
            self.Ecgn_y = np.load(os.path.join(self.CMG_coord_folder, 'Etchegoin_y_coord.npy'))
            self.Ecgn_z = np.load(os.path.join(self.CMG_coord_folder, 'Etchegoin_z_coord.npy'))
        else:
            print('sys.platform type not implemented yet. Please check!!!')


    def save_dict_of_npy_from_pkl(self, file_name, data):
        """
        file_name is a str with .pkl. E.g.: 'test.pkl'
        """
        with open(file_name, 'wb') as output:
            # Pickle dictionary using protocol 0.
            pickle.dump(data, output)

    def load_dict_of_npy_from_pkl(self, file_name):
        """
        file_name is a str with .pkl. E.g.: 'test.pkl'
        """
        with open(file_name, "rb") as fp:
            data = pickle.load(fp)
        return data
    
    def calc_4D_prop_change(self, data, dimt):
        """
        Goal: calculate the property change in 4D arrays. The shape and index sequence do not change.
        Input:
        1. Data: 4d array.
        2. dimt: the dimension index for time axis.
        Output:
        1. dp_data: 4d array to show the change of property versus the initial. 
        """
        num_t_steps = data.shape[dimt]
        # Select the dimt dimension, pick the 0 idx as initial
        # Reduced 1 dimension
        idx = [slice(None)]*data.ndim
        idx[dimt] = 0
        data_init = data[tuple(idx)]

        # Expend 1 dimension
        idx_expand = [slice(None)]*data.ndim
        idx_expand[dimt] = np.newaxis
        data_init_to_expand = data_init[tuple(idx_expand)]

        # Copy the initial T times along this newly expand dimension
        data_init_expand = np.repeat(data_init_to_expand, num_t_steps, axis=dimt)

        # Calculate the dP along all time steps
        dp_data = data - data_init_expand
        
        return dp_data


    def rescale_3d_images(self, data_3d, tdim, rawX=None, rawY=None):
        """
        Goal: rescale the 3d npy in x,y plane. 
        Input
        1. data_3d: input data for rescale. 
                    Example: (33,33,120) or (num_xx, num_yy, num_tt)
        2. tdim: scalar. Define the dimension to unfold.
        Output
        1. result_3d: rescaled 3d data. The shape has been updated to: (num_tt, rescaleX_num_grids, rescaleY_num_grids)
                    Example: (120,200,200)
        """
        # slice all dimensions
        idx = [slice(None)]*data_3d.ndim
        num_t = data_3d.shape[tdim]
        rst = []
        for tt in range(num_t):
            # Define which dimension to slice, and the id among that dimension to unfold from 3d to 2d.
            idx[tdim] = tt
            rescaled_data_temp = self.rescale_ijk2xyz(X_2d=data_3d[tuple(idx)], rawX=rawX, rawY=rawY)
            rst.append(rescaled_data_temp)
        result_3d = np.array(rst)
        return result_3d

    def rescale_4d_images_in_loop(self, path2folder, case_names, dimz, dimz_idx, dimt):
        """
        Goal: automatically rescale 3d images in batch/loop and return all npy in a dict.
        Input
        1. path2folder: path to the folder of all npy files.
        2. case_names:  a list of str with extension .npy.
                        example: ['case0_VERDSPLGEO.npy', 'case1_VERDSPLGEO.npy', ..., 'case100_VERDSPLGEO]
        3. dimz: a scalar value to define the dimension of z to select in the 4d array.
                        example: dimz = 2 for 4d array of (33,33,75,120) or (num_xx, num_yy, num_zz, num_tt)
        4. dimz_idx: int or str. select the idx of dimz to reduce the npy array from 4d to 3d.
                        example: dimz_idx = 0 to select the top layer (land surface).
                                dimz_idx = 'max' to calculate the max along this axis.
                                dimz_idx = 'dpmax' to calculate the max of property change along this axis.
        5. dimt: a scalar value to define dimension of t to select in the 4d array.
                        example: dimt = 3 for 4d array of (33,33,75,120) or (num_xx, num_yy, num_zz, num_tt)
                                 b/c data[tuple(idx)] already reduces 1 dimension.
        Output:

        """
        rst = {}
        for ii, casename in enumerate(tqdm(case_names)):
            try:
                data = np.load(os.path.join(path2folder, casename))
                if isinstance(dimz_idx,int):
                    idx = [slice(None)]*data.ndim
                    idx[dimz] = dimz_idx
                    data_tobe_rescaled = data[tuple(idx)]
                    # The selection action reduces 1 dimension. Update dimt
                    if dimt > dimz:
                        dimt -= 1
                elif isinstance(dimz_idx,str):
                    if dimz_idx == 'max':
                        data_tobe_rescaled = np.max(data, axis=dimz)
                        # The np.max action reduces 1 dimension. Update dimt
                        if dimt > dimz:
                            dimt -= 1
                    elif dimz_idx == 'dpmax':
                        # Calculate the dP 4D array
                        dp_data = self.calc_4D_prop_change(data=data, dimt=dimt)
                        # Get the max along an axis
                        data_tobe_rescaled = np.max(dp_data, axis=dimz)
                        # The np.max action reduces 1 dimension. Update dimt
                        if dimt > dimz:
                            dimt -= 1
                    else:
                        print(f'{casename}: dimz_idx value of {dimz_idx} is not implemented in rescale_4d_images_in_loop function of pycmgpostcalc class in pyCMG_Results.py ...')
                else:
                    print(f'{casename}: dimz_idx of {dimz_idx} is not str or int type in rescale_4d_images_in_loop function of pycmgpostcalc class in pyCMG_Results.py ...')
                data3d_new = self.rescale_3d_images(data_tobe_rescaled, tdim=dimt)
                rst[casename.split('.')[0]] = data3d_new
            except:
                print(f'{casename} is not saved in results due to an error of reading or rescaling of 4d array (check data exists or not) ...')
        return rst
    

    def interpolate_1d(self, x, y, newx):
        """
        Goal: 1d interpolation function.
        """
        f = interpolate.interp1d(x, y)
        newy = f(newx) 
        return newy


    def rescale_ijk2xyz(self, X_2d, rawX=None, rawY=None):
        """
        Goal: rescale the 2d matrix from ijk (simulation) coordinate to xyz (real world)
        Input
        1. X_2d:            the property that need to be scaled.
        2. num_new_coord:   number of grids or resolution of new coordinate system
        3. rawX:            raw coordinate of X. 1d array with ascending order. Unique values required.
        4. rawY:            raw coordinate of Y. 1d array with ascending order. Unique values required.
        Output
        1. newX_2d:         rescaled property in 2d but interpolate to new grids.
        # 2. newX:            2D array (shape is num_new_coord x num_new_coord). New coordinate of X.
        # 3. newY:            2D array (shape is num_new_coord x num_new_coord). New coordinate of Y.
        """


        # num_xx, num_yy = X_2d.shape

        if rawX is not None:
            xmin = rawX.min()
            xmax = rawX.max()
            X = rawX
            # X = np.linspace(xmin,xmax,num_xx)
            # X = self.interpolate_1d(x=np.arange(1,int(self.Ecgn_x[:,0,0].shape[0]+1)),
            #                         y=rawX,
            #                         newx=np.arange(1,int(num_xx+1)))
            
        else:
            # Load Etchegoin related data
            self.load_Etchegoin_6x6_data()
            
            xmin = self.Ecgn_x[:,:,0].min()
            xmax = self.Ecgn_x[:,:,0].max()
            X = self.Ecgn_x[:,0,0]
            # X = self.interpolate_1d(x=np.arange(1,int(self.Ecgn_x[:,0,0].shape[0]+1)),
            #                         y=self.Ecgn_x[:,0,0],
            #                         newx=np.arange(1,int(num_xx+1)))

        if rawY is not None:
            ymin = rawY.min()
            ymax = rawY.max()
            Y = rawY
            # Y = self.interpolate_1d(x=np.arange(1,int(rawY.shape[0]+1)),
            #                         y=rawY,
            #                         newx=np.arange(1,int(num_yy+1)))
        else:
            ymin = self.Ecgn_y[:,:,0].min()
            ymax = self.Ecgn_y[:,:,0].max()
            Y = self.Ecgn_y[0,:,0]
            # Y = self.interpolate_1d(x=np.arange(1,int(self.Ecgn_y[0,:,0].shape[0]+1)),
            #                         y=self.Ecgn_y[0,:,0],
            #                         newx=np.arange(1,int(num_yy+1)))


        xx = np.linspace(xmin,xmax,self.rescaleX_num_grids)
        yy = np.linspace(ymin,ymax,self.rescaleY_num_grids)
        self.newX_1d = xx
        self.newY_1d = yy
        newX, newY = np.meshgrid(xx, yy, indexing='ij')

        interp = RegularGridInterpolator((X, Y), X_2d, method=self.rescale_method, bounds_error=False, fill_value=None)
        newX_2d = interp((newX, newY))

        return newX_2d



    def calc_CO2plume(self, fourDarr, time_query, co2threshold, prop, layer_nums, sizemul=None):
        """
        This only includes the maximum land co2 plume method.
        Goal: Compute plume size in area at land surface, and plume shape profiles.
        Inputs:
        1. fourDarr: 4d array (xx,yy,zz,tt). 3D model property change with time.
        2. co2threshold: a constant value or 2d array (shape is num_new_coord by num_new_coord). Define threhold to define the plume boundary.
        3. sizemul: if None, plumesize shows the ratio of grids inside of the CO2 plume boundary to all grids.
                    if Not None, sizemul is the total area of the model with an unit in area. plumesize shows the CO2 plume size.
        Outputs:
        1. plumesize: ratio or actual size, depending on sizemul.
        2. all_points: 
        3. all_outlines
        """
        
        # combo_SG = pycmgresults.rwodf2arr(globals()[f'df{prop}_lyr{layer_nums[0]}'], query=time_query)
        # m,n = combo_SG.shape
        # for ll in layer_nums:
        #     globals()[f'arr{prop}_lyr{ll}'] = pycmgresults.rwodf2arr(globals()[f'df{prop}_lyr{ll}'], query=time_query)
        #     combo_SG = np.stack((combo_SG,globals()[f'arr{prop}_lyr{ll}']), axis=2)

        # max_combo_SG = np.max(combo_SG, axis=2)

        n,m,z,t = fourDarr.shape
        max_combo_SG = np.max(fourDarr, axis=2)
        # rescale 2d array with new coordinates
        rescale_max_combo_SG = self.rescale_ijk2xyz(X_2d=max_combo_SG)
        # Compare with the threshold
        plume = rescale_max_combo_SG > co2threshold
        if sizemul is None:
            plumesize = np.sum(plume,axis=(0,1))/self.rescaleX_num_grids/self.rescaleY_num_grids
            print('Plume size is ratio of num of grids with CO2 to total num of grids, wtihout a sizemul defined ...')
        else:
            plumesize = np.sum(plume,axis=(0,1))/self.rescaleX_num_grids/self.rescaleY_num_grids*sizemul

        all_points, all_outlines = [], []
        newX, newY = np.meshgrid(self.newX_1d, self.newY_1d, indexing='ij')
        for tt in range(t):
            ########################################################################
            # The following has a problem about x y index because the shape rotates
            pts, outline = self.plume_outline(plume[:,:,tt])
            all_points.append(pts)
            # all_outlines.append(outline)
            ########################################################################
            shape_idx = self.outline_idx_from_2D_TrueFalse_array(x=plume[:,:,tt], flag=True)
            shape_xy = self.ijindex_to_XYcoord(shape_idx, newX, newY)
            all_outlines.append(shape_xy)

        # _, globals()[f'co2outline_{time_query}'] = self.plume_outline(plume)
        return plumesize,all_points,all_outlines
    
    def calc_plume_size(self, arr4D, dim_max, threshold, sizemul=None):
        """
        Goal: Compute plume size in area at land surface, and plume shape profiles. arr4D is upscaled or downscaled.
        Inputs:
        1. arr4D: 4d array (xx,yy,zz,tt). 3D model property change with time.
        2. dim_max: the dimension of arr4D to be max.
        3. threshold: a constant value or 3d array (shape is num_new_coord by num_new_coord). Define threhold to define the plume boundary.
        4. sizemul: if None, plumesize shows the ratio of grids inside of the CO2 plume boundary to all grids.
                    if Not None, sizemul is the total area of the model with an unit in area. plumesize shows the CO2 plume size.
        Outputs:
        1. plumesize: ratio or actual size, depending on sizemul.
        2. all_outlines: a list of lists with x,y coordinate (not i,j coordinate)
        """
        n,m,z,t = arr4D.shape
        max_combo_SG = np.max(arr4D, axis=dim_max)
        result_3d = self.rescale_3d_images(data_3d=max_combo_SG, tdim=2, rawX=None, rawY=None)
        plume = result_3d > threshold
        if sizemul is None:
            plumesize = np.sum(plume,axis=(1,2))/self.rescaleX_num_grids/self.rescaleY_num_grids
            print('Plume size is ratio of num of grids with CO2 to total num of grids, wtihout a sizemul defined ...')
        else:
            plumesize = np.sum(plume,axis=(1,2))/self.rescaleX_num_grids/self.rescaleY_num_grids*sizemul

        all_points, all_outlines = [], []
        newX, newY = np.meshgrid(self.newX_1d, self.newY_1d, indexing='ij')
        for tt in range(t):
            shape_idx = self.outline_idx_from_2D_TrueFalse_array(x=plume[tt,:,:], flag=True)
            shape_xy = self.ijindex_to_XYcoord(shape_idx, newX, newY, type='outline')
            all_outlines.append(shape_xy)
            pts_xy = self.ijindex_to_XYcoord(shape_idx, newX, newY, type='pts')
            all_points.append(pts_xy)

        return plumesize, all_points, all_outlines
    

    def calc_2D_image_plume_size(self, arr2D, threshold_val, sizemul=None):
        """
        Goal: compute the plume size given a threshold for 2D input image data.
        Input: arr2D in shape (dimx, dimy). 
                threshold_val: could be a float value or a 2D matrix.
        Output: a single value of normalized plume size. 
                Normalized plume size = num of grids in plume / num of total grids. 
                Scale to query points with equal space.
        """

        # Rescale
        arr2D_scale = self.rescale_ijk2xyz(X_2d=arr2D)
        plume2D = arr2D_scale > threshold_val
        plume2D_size = np.sum(plume2D)/self.rescaleX_num_grids/self.rescaleY_num_grids

        if sizemul:
            rst = plume2D_size*sizemul
        else:
            rst = plume2D_size

        return rst
    

    def calc_2D_image_plume_size_in_loop(self, arr4D, threshold_val, sizemul=None):
        """
        Goal: compute plume size in loop for 4D data in a more easy way.
        Input: arr4D in shape (num_sample, num_t, num_x, num_y)
                threshold_val: could be a float value or a 2D matrix.
        Output: array of plume size at different time steps for different cases.
                Shape (num_sample, num_time steps)
        """
        rst = []
        num_sample, num_t, num_x, num_y = arr4D.shape

        for nn in range(num_sample):
            rst_sample = []
            for tt in range(num_t):
                arr2D = arr4D[nn, tt, :, :]
                plumesize = self.calc_2D_image_plume_size(arr2D, threshold_val, sizemul)
                rst_sample.append(plumesize)
            rst.append(rst_sample)

        return np.array(rst)





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
    
    def calc_plume_edge_dist2inj(self, pts, inj, X, Y):
        dist = []
        norm_dist_params = []
        bound_dist = []
        t = len(pts)
        xy_inj = (self.ijk2xyz(inj[0],X), self.ijk2xyz(inj[1],Y))
        for tt in range(t):
            if len(pts[tt]) > 1:
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
            else:
                # Case when t=0
                dist.append(0)
                norm_dist_params.append((0,0))
                bound_dist.append((0,0))
        return dist, norm_dist_params, bound_dist
    

    def calc_plume_dist2inj(self, allpts, inj_xy=(1528843.31941562, 683135.98292623)):
        """
        Goal: calculate plume edge distance to injector. Unit: ft
        Inputs:
        1. allpts: list of lists. len(allpts) = num_time_steps. len(allpts[t]) = num_pts_at_the_edge. All pts units are ft in xy coordinate.
        2. inj_xy: injector XY coordinate at land surface.
        Outputs:
        1. dist: list of lists. len(dist) = num_time_steps. len(dist[t]) = num_pts_at_the_edge. Record all distances.
        2. norm_dist_params: list of tuples (mu, std). len(norm_dist_params) = num_time_steps. Fit with normal distn.
        3. bound_dist: list of tuples (min, max). len(norm_dist_params) = num_time_steps. Record the wide ranges.
        """

        dist = []
        norm_dist_params = []
        bound_dist = []

        for tt, pt_t in enumerate(allpts):
            if len(pt_t) > 1:
                dd = []
                for pp in pt_t:
                    dd.append(self.calc_dist(A=inj_xy,B=pp))
                # dd collects all pts_xy at a time step
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
    
    def load_injection_horizon_list(self, inj_hrzn_list):
        """
        len of inj_hrzn_list = len of caseid
        inj_hrzn_list[ii] = inj_hrzn[ii] in dfexp 
        e.g.: for cases with the same injection profile, inj_hrzn_list = [18,...,18]
        """
        self.num_time_steps = inj_hrzn_list


    def combine_verdispgeo_npy(self, path2folder, caseids, calc_type='raw'):
        """
        Input data: (num_xx, num_yy, num_zz, num_tt). E.g.: (33, 33, 2, 120)
        Output data: (num_samples, num_tt, num_xx, num_yy) E.g.: (810, 18, 33, 33)
        """

        rst = []
        for ii, cid in enumerate(tqdm(caseids)):
            cname = f"case{cid}_VERDSPLGEO.npy"
            arr = np.load(os.path.join(path2folder, cname))
            if calc_type.lower() == 'raw':
                arr_select = arr[:,:,0,1:int(self.num_time_steps[ii]+1)]
            elif calc_type.lower() == 'diff':
                arr_select = np.diff(arr[:,:,0,0:int(self.num_time_steps[ii]+1)], n=1, axis=2)
            else:
                print(f"calc_type {calc_type} not implemented in combine_verdispgeo_npy function ...")
            rst.append(np.transpose(arr_select, (2, 0, 1)))
        rst_arr = np.array(rst)
        return rst_arr
    

    def combine_SG_npy(self, path2folder, caseids, calc_type='maxProjection', threshold=0.05):
        """
        Input data: (num_xx, num_yy, num_zz, num_tt). E.g.: (33, 33, 75, 120)
        Output data: (num_samples, num_tt, num_xx, num_yy) E.g.: (810, 18, 33, 33)
        """
        rst = []
        for ii, cid in enumerate(tqdm(caseids)):
            cname = f"case{cid}_SG.npy"
            arr = np.load(os.path.join(path2folder, cname))
            if calc_type.lower() == 'maxprojection':
                arr_max = np.max(arr, axis=2)
                arr_select = arr_max[:,:,1:int(self.num_time_steps[ii]+1)]
            elif calc_type.lower() == 'binplume':
                plume = (arr > threshold)*1
                arr_max = np.max(plume, axis=2)
                arr_select = arr_max[:,:,1:int(self.num_time_steps[ii]+1)]
            else:
                print(f"calc_type {calc_type} not implemented in combine_SG_npy function ...")
            rst.append(np.transpose(arr_select, (2, 0, 1)))
        rst_arr = np.array(rst)
        return rst_arr
    
    def combine_PRES_npy(self, path2folder, caseids, calc_type='maxProjection', threshold=145.038):
        """
        Input data: (num_xx, num_yy, num_zz, num_tt). E.g.: (33, 33, 75, 120)
        Output data: (num_samples, num_tt, num_xx, num_yy) E.g.: (810, 18, 33, 33)
        """
        rst = []
        for ii, cid in enumerate(tqdm(caseids)):
            cname = f"case{cid}_PRES.npy"
            arr = np.load(os.path.join(path2folder, cname))
            darr = self.calc_4D_prop_change(data=arr, dimt=3)
            
            if calc_type.lower() == 'maxprojection':
                arr_max = np.max(darr, axis=2)
                arr_select = arr_max[:,:,1:int(self.num_time_steps[ii]+1)]
            elif calc_type.lower() == 'binplume':
                plume = (darr > threshold)*1
                arr_max = np.max(plume, axis=2)
                arr_select = arr_max[:,:,1:int(self.num_time_steps[ii]+1)]
            else:
                print(f"calc_type {calc_type} not implemented in combine_PRES_npy function ...")
            rst.append(np.transpose(arr_select, (2, 0, 1)))
        rst_arr = np.array(rst)
        return rst_arr


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

    def ijindex_to_XYcoord(self, shape_idx, matX, matY, type='outline'):
        """
        shape_idx: list of tuples [(i1,ji),...,(in,jn)]. Index of boundary points.
        matX: 2D matrix array for X coordinates in ft or lat for all grids
        matY: 2D matrix array for Y coordinates in ft or long for all grids
        """
        plot_pts = []
        for ii, idx in enumerate(shape_idx):

            pt1x = matX[idx]
            pt1y = matY[idx]

            if type=='outline':

                pt2x = matX[shape_idx[ii-1]]
                pt2y = matY[shape_idx[ii-1]]

                plot_pts.append(([pt1x, pt2x],[pt1y, pt2y]))
            elif type=='pts':
                plot_pts.append([pt1x, pt1y])
        
        return plot_pts
    
