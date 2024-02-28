'''
Author: Yunan Li
Date: 2023/05/17
Goal: This file contains class and functions to create CMG model dat files in Python
Class:  1. well_design_opt: well design optimization
'''

import numpy as np
import pandas as pd
import os

import sys
# append the path of the parent directory
sys.path.append("..")

# import class of the base case file
from cmg_models.wrtcmgdat_SPR_CCS_field6x6 import Write_datfiles_SPRCCS
from cmg_models.wrtcmgdat_h2_rxns import Write_datfiles_h2


class well_design_opt():
    def __init__(self):
        super().__init__()
        self.title1 = 'Well opt design exp3'
        self.title2 = 'no BHP constraint'
        self.title3 = 'Run by Yunan Li'
    
    def create_3Dwell_path(self, initial, end):
        """
        Goal: create the well trajectory to define the perforation grids given the initial point and end point coordinates
        Initial and end params should be in format of (ix,iy,iz)
        """
        ix0, iy0, iz0 = initial
        ix, iy, iz = end
        
        num_ix = np.abs(ix-ix0)+1
        num_iy = np.abs(iy-iy0)+1
        
        # Decide the direction from initial to end points
        if ix >= ix0:
            ssx = 2
        else:
            ssx = 1
        if iy >= iy0:
            ssy = 2
        else:
            ssy = 1
        if iz >= iz0:
            ssz = 2
        else:
            ssz = 1
        
        # For cases ix==ix0, iy==iy0
        if ix==ix0 and iy==iy0:
            num_iz = np.abs(iz-iz0)+1
            ix_pts = [ix0]*num_iz
            iy_pts = [iy0]*num_iz
            iz_pts = [iz0+kk*(-1)**ssz for kk in range(num_iz)]
            
        else:
            # For the case ix==ix0, xy_slope edge case
            if ix == ix0:
                ix_pts = [ix0]*num_iy
                iy_pts = [iy0+jj*(-1)**ssy for jj in range(num_iy)]

            else:
                xy_slope = (iy-iy0)/(ix-ix0)
                if num_ix > num_iy:
                    ix_pts = [ix0+ii*(-1)**ssx for ii in range(num_ix)]
                    iy_pts = [int(np.round(iy0+xy_slope*(ix_pts[ii]-ix0))) for ii in range(num_ix)]
                elif num_ix < num_iy:
                    iy_pts = [iy0+jj*(-1)**ssy for jj in range(num_iy)]
                    ix_pts = [int(np.round((iy_pts[jj]-iy0)/xy_slope+ix0)) for jj in range(num_iy)]
                else:
                    ix_pts = [ix0+ii*(-1)**ssx for ii in range(num_ix)]
                    iy_pts = [iy0+jj*(-1)**ssy for jj in range(num_iy)]
            
            # Get the z direction coordinates
            z_slope = (iz-iz0)/max(num_ix,num_iy)
            iz_pts = [int(np.round(iz0+z_slope*kk)) for kk in range(max(num_ix,num_iy))]
            
        return ix_pts, iy_pts, iz_pts
    
    def write_dat_file(self, folder_path, dat_file_name, initial_pt, end_pt, caseid=None):
        """
        Goal: write dat files based on the ref base CMG model.
        Return: xyz coord of well perforation grids in 3 lists (They are of the same length)
        pts are in format of tuple (ix,iy,iz)
        """
        # Create the instance
        wrtdat = Write_datfiles_SPRCCS()
        # Create a folder if not exists
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if caseid is None:
            caseid = 'N/A'
        # Define the new file name and path
        newfile_path = os.path.join(folder_path, dat_file_name)
        filename = open(newfile_path, 'w+')
        wrtdat.print_heading(self.title1,self.title2,self.title3,f'case{caseid}',fileID=filename)
        wrtdat.print_before_well(fileID=filename)
        xpts, ypts, zpts = self.create_3Dwell_path(initial_pt, end_pt)
        wrtdat.print_well_config(ix=xpts, iy=ypts, iz=zpts, fileID=filename)
        wrtdat.print_after_well(fileID=filename)
        filename.close()
        return xpts, ypts, zpts
    

    def write_simfiles(self, folder_path, initial_pt, d_ix, d_iy, d_iz, caseid, csv_name=None):
        """
        Well design optimization
        Goal: create the well design params space and write all scenarios in cmg dat files format. 
        folder_path: folder that all cmg dat files will be saved in
        initial_pt: (ix,iy,iz) tuple format. Doc the initial point i,j,k coord in simulation model.
        d_ix: [list] list of integers b/c coord are ints. Each item shows the dist. to the end point along x direction.
        d_iy: [list] list of integers b/c coord are ints. Each item shows the dist. to the end point along x direction.
        d_iz: [list] list of integers b/c coord are ints. Each item shows the dist. to the end point along x direction.
        caseid: The starting case ID number
        csv_name: name of the csv file to doc the exp. design. If None, no csv file will be created.
        """
        CASEID_list = []
        caseid0 = caseid
        ix0,iy0,iz0 = initial_pt
        if csv_name is not None:
            dfexp = pd.DataFrame({})
            xx,yy,zz = [],[],[]
        # Create params space
        for kk in d_iz:
            for jj in d_iy:
                for ii in d_ix:
                    # if ii!=0 or jj!=0:
                    end_pt = (int(ix0+ii), int(iy0+jj), int(iz0+kk))
                    dat_file_name = f'case{caseid}.dat'
                    xpts, ypts, zpts = self.write_dat_file(folder_path, dat_file_name, initial_pt, end_pt, caseid)
                    CASEID_list.append(caseid)

                    if csv_name is not None:
#                        dat.append(f'case{caseid}')
                        xx.append(xpts)
                        yy.append(ypts)
                        zz.append(zpts)
                    # Count the next case id
                    caseid += 1
        if csv_name is not None:
            dfexp['CaseID'] = [f'case{i}' for i in CASEID_list]
            dfexp['Xcoord'] = xx
            dfexp['Ycoord'] = yy
            dfexp['Zcoord'] = zz
            dfexp.to_csv(os.path.join(os.path.dirname(folder_path), csv_name),index=False)
            print(f'There are {caseid-caseid0} of CMG simulation cases created in the assigned folder ...')
        
        return CASEID_list

        
    def read_well_design_opt_csv(self, folder_path, csv_name):
        """
        Goal: convert the str of xyz coord lists back to list in the df.
        """
        dfread = pd.read_csv(os.path.join(folder_path, csv_name))
        dfread['Xcoord'] = dfread['Xcoord'].agg(lambda x: eval(x))
        dfread['Ycoord'] = dfread['Ycoord'].agg(lambda x: eval(x))
        dfread['Zcoord'] = dfread['Zcoord'].agg(lambda x: eval(x))
        return dfread



class global_sa():
    def __init__(self):
        super().__init__()
        self.title1 = 'Global sensitivity'
        self.title2 = 'InSAR'
        self.title3 = 'Run by Yunan Li'
        self.yr_after_shutin = [5, 10]

    def write_dat_file(self, folder_path, dat_file_name, rock_mech, rsvr, caseid=None):
        """
        Goal: write dat files based on the ref base CMG model.
        Return: N/A
        """
        # Create the instance
        wrtdat = Write_datfiles_SPRCCS()
        wrtdat.yr_after_shutin = self.yr_after_shutin
        # Create a folder if not exists
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if caseid is None:
            caseid = 'N/A'
        # Define the new file name and path
        newfile_path = os.path.join(folder_path, dat_file_name)
        filename = open(newfile_path, 'w+')
        wrtdat.print_heading(self.title1,self.title2,self.title3,f'case{caseid}',fileID=filename)
        wrtdat.print_before_mech(rsvr, fileID=filename)
        wrtdat.print_gmech(rock_mech, fileID=filename)
        wrtdat.print_well_control(rsvr, fileID=filename)
        filename.close()

    def write_simfiles(self, folder_path, df, verbose=True):
        
        n, num_col = df.shape
        rock_mech = {}
        rsvr = {}
        for i in range(n):
            dftemp = df.iloc[i]
            caseid = dftemp['caseid']
            dat_file_name = f'case{int(caseid)}.dat'
            rock_mech['E_shale'] = dftemp['E_shale, psi']
            rock_mech['E_sand'] = dftemp['E_sand, psi']
            rock_mech['v_shale'] = dftemp['v_shale']
            rock_mech['v_sand'] = dftemp['v_sand']

            rsvr['kvkh'] = dftemp['kvkh']
            rsvr['inj_rate'] = dftemp['inj_rate, ft3/day']
            rsvr['inj_hrzn'] = dftemp['inj_hrzn, year']

            self.write_dat_file(folder_path, dat_file_name, rock_mech, rsvr, caseid=caseid)

        if verbose:
            print('All cases from the df are written into CMG dat file format ...')


class reaction_h2():
    def __init__(self):
        super().__init__()
        self.title1 = 'H2 ISCG kinetics (case1)'
        self.title2 = 'HM'
        self.title3 = 'Run by Yunan Li'


    def write_dat_file(self, folder_path, dat_file_name, df_params, caseid=None):
        """
        Goal: write single dat file based on the df_params and ref base CMG model.
        Return: N/A
        """
        # Create the instance
        wrtdat = Write_datfiles_h2()
        wrtdat.params = df_params
        # Create a folder if not exists
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if caseid is None:
            caseid = 'N/A'
        # Define the new file name and path
        newfile_path = os.path.join(folder_path, dat_file_name)
        filename = open(newfile_path, 'w+')
        wrtdat.print_gas_case1(fileID=filename)
        filename.close()
