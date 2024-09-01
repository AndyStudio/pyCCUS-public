'''
Author: Yunan Li
Date: 2023/11/26
Goal: This file contains a series of components to interact with CMG.
Class:  1. pycmgcontrol
'''

import numpy as np
import pandas as pd
import os
import shutil

import sys
# append the path of the parent directory
sys.path.append("..")
# import method from sibling module
from utils.pyCMG_Simulator import wrt_cmgrwd_grids
from utils.pyCMG_Results import pycmgresults

class pycmgcontrol():
    def __init__(self, exp_name, simfolder):
        super().__init__()
        self.exp_name = exp_name
        self.simfolder = simfolder
        self.batchfolder = ''
        self.proplist = ['SG','PRES']
        self.layer_nums=np.arange(75)+41
        self.time_start_year = 2023
        self.time_query = [self.time_start_year+i for i in range(0,120)]
        self.rwd_time_step='*ALL-TIMES'
        self.rwd_layer_type = '*XYZLAYER'
        self.rwd_precis = 4
        ##### Global SA params
        # self.inj_hrzn = None
        
        # System selection
        self.cmg_version = 'ese-win32-v2022.30'
        self.err_stop = False
        ##################################################
        ##### DON't manually update params in this section
        ##### Empty ...
        self.cmg2npy = None
        ##################################################
        # self.yr_after_shutin_disp = [5, 10]
        ##################################################
        ##### Params to control rwo2npy steps ######
        self.XY2arr_interp_method = "cubic"  # options = {‘linear’, ‘nearest’, ‘cubic’}
        self.XY2arr_interp_num_x = 100
        self.XY2arr_interp_num_y = 100
        self.x_dir_key = 'X'
        self.y_dir_key = 'Y'
        ##################################################




    def run_stars_simulation(self, case_name_suffix):
        
        if self.cmg_version == 'ese-win32-v2022.30':
            exe_path='"C:\\Program Files (x86)\\CMG\\STARS\\2022.30\\Win_x64\\EXE\\st202230.exe"'
            cd_path = os.path.join(self.simfolder, self.batchfolder).rstrip('\\')

        elif self.cmg_version == 'ese-ts1win-v2023.20':
            exe_path='"C:\\Program Files\\CMG\\STARS\\2023.20\\Win_x64\\EXE\\st202320.exe"'
            cd_path = os.path.join(self.simfolder, self.batchfolder).rstrip('\\')
                
        elif self.cmg_version == 'stf-sherlock-v2020.10':
            exe_path = "/home/groups/s-ees/share/cees/software/x86_64_arch/CMG/2020.109/stars/2020.11/linux_x64/exe/st202011.exe" 
            cd_path = os.path.join(self.simfolder, self.batchfolder).rstrip('/')
        else:
            if self.err_stop:
                raise ValueError(f'The CMG version {self.cmg_version} is not implemented in run_gem_simulation function of pycmgcontrol() class .....')
            else:
                print(f'The CMG version {self.cmg_version} is not implemented in run_gem_simulation function of pycmgcontrol() class .....')
            
        
        # Execute the CMG software using the files we defined.
        # cmd_line = 'cd ' + cd_path + '  & ' + exe_path + '  -f ' + f'{case_name_suffix}'
        cmd_line = f"cd {cd_path}  & {exe_path} -f {case_name_suffix}"
        try:
            os.system(cmd_line)
        except:
            if self.err_stop:
                raise ValueError(f'{case_name_suffix} run CMG STARS step encounters an error ...')
            else:
                print(f'{case_name_suffix} run CMG STARS step encounters an error ...')
    

    def run_gem_simulation(self, case_name_suffix):
        
        if self.cmg_version == 'ese-win32-v2022.30':
            exe_path='"C:\\Program Files (x86)\\CMG\\GEM\\2022.30\\Win_x64\\EXE\\gm202230.exe"'
            cd_path = os.path.join(self.simfolder, self.batchfolder).rstrip('\\')

        elif self.cmg_version == 'ese-ts1win-v2023.20':
            exe_path='"C:\\Program Files\\CMG\\GEM\\2023.20\\Win_x64\\EXE\\gm202320.exe"'
            cd_path = os.path.join(self.simfolder, self.batchfolder).rstrip('\\')
                
        elif self.cmg_version == 'stf-sherlock-v2020.10':
            exe_path = "/home/groups/s-ees/share/cees/software/x86_64_arch/CMG/2020.109/gem/2020.11/linux_x64/exe/gm202011.exe" 
            cd_path = os.path.join(self.simfolder, self.batchfolder).rstrip('/')
        else:
            if self.err_stop:
                raise ValueError(f'The CMG version {self.cmg_version} is not implemented in run_gem_simulation function of pycmgcontrol() class .....')
            else:
                print(f'The CMG version {self.cmg_version} is not implemented in run_gem_simulation function of pycmgcontrol() class .....')
            
        
        # Execute the CMG software using the files we defined.
        # cmd_line = 'cd ' + cd_path + '  & ' + exe_path + '  -f ' + f'{case_name_suffix}'
        cmd_line = f"cd {cd_path}  & {exe_path} -f {case_name_suffix}"
        try:
            os.system(cmd_line)
        except:
            if self.err_stop:
                raise ValueError(f'{case_name_suffix} run CMG GEM step encounters an error ...')
            else:
                print(f'{case_name_suffix} run CMG GEM step encounters an error ...')

    def wrt_rwd_report(self, case_name, verbose=False):
            # write cmg rwd file to simfolder
        wrtfolder = os.path.join(self.simfolder, self.batchfolder)
        wrt_cmgrwd_grids(sim_sr3=case_name,
                        ext_rwo=None,
                        path2rwd=wrtfolder,
                        create_rwo_folder=True,
                        rwo_folder=f'rwo_{case_name}',
                        proplist=self.proplist,
                        layer_num=self.layer_nums,
                        time_step=self.rwd_time_step,
                        layer_type=self.rwd_layer_type,
                        precis=self.rwd_precis,
                        cmg_version=self.cmg_version,
                        verbose=verbose)
        
    def folder_sanity_check(self, case_name):
        # Check if rwo folder already created. If not, we create
        self.rwo_folder = os.path.join(self.simfolder, self.batchfolder, f'rwo_{case_name}')
        if not os.path.exists(self.rwo_folder):
            os.makedirs(self.rwo_folder)
        
        self.npy_folder = os.path.join(self.simfolder, self.batchfolder, 'rst_npy')
        if not os.path.exists(self.npy_folder):
            os.makedirs(self.npy_folder)

    def run_rwd_report(self, case_name):
        """
        Files folder path template
        exe_path='"C:\\Program Files (x86)\\CMG\\RESULTS\\2022.30\\Win_x64\\EXE\\Report.exe"'
        cd_path='E:\\CUSP_win\\GEM_CCS\\SPR_model\\SPR_petrel_model2CMG\\extended_6x6\\Etchegoin_shale\\well_design_exp3_VOLMOD2 '
        """
   
        if self.cmg_version == 'ese-win32-v2022.30':
            exe_path='"C:\\Program Files (x86)\\CMG\\RESULTS\\2022.30\\Win_x64\\EXE\\Report.exe"'
            cd_path = os.path.join(self.simfolder, self.batchfolder).rstrip('\\')
    #         if batchfolder == 'N/A':
    #             cd_path = simfolder.rstrip('\\')
    #         else:
    #             cd_path = os.path.join(simfolder, batchfolder)
        elif self.cmg_version == 'ese-ts1win-v2023.20':
            exe_path='"C:\\Program Files\\CMG\\RESULTS\\2023.20\\Win_x64\\exe\\Report.exe"'
            cd_path = os.path.join(self.simfolder, self.batchfolder).rstrip('\\')
                
        elif self.cmg_version == 'stf-sherlock-v2020.10':
            exe_path = "/home/groups/s-ees/share/cees/software/x86_64_arch/CMG/2020.109/gem/2020.11/linux_x64/exe/gm202011.exe" 
            cd_path = os.path.join(self.simfolder, self.batchfolder).rstrip('/')
        else:
            if self.err_stop:
                raise ValueError(f'The CMG version {self.cmg_version} is not implemented in run_rwd_report function of pycmgcontrol() class .....')
            else:
                print(f'The CMG version {self.cmg_version} is not implemented in run_rwd_report function of pycmgcontrol() class .....')
            
        
        # Execute the CMG software using the files we defined.
        # cmd_line = 'cd ' + cd_path + '  & ' + exe_path + '  -f ' + f'{case_name}' + '.rwd'
        # work_version = 'cd ' + cd_path + ' & ' + exe_path + '  -f ' + f'"case{case_name}"' + '.rwd'
        cmd_line = f"cd {cd_path}  & {exe_path} -f {case_name}.rwd"
        try:
            os.system(cmd_line)
        except:
            if self.err_stop:
                raise ValueError(f'{case_name} run rwd step encounters an error ...')
            else:
                print(f'{case_name} run rwd step encounters an error ...')

    def read_PRES_SG_from_rwo(self, case_name):
        # Read SG and PRES arrays
        self.SG_flag = self.read_SG_rwo2npy(case_name=case_name, save=True)
        SG_temp = self.cmg2npy
        self.PRES_flag = self.read_PRES_rwo2npy(case_name=case_name, save=True)
        self.cmg2npy = [SG_temp, self.cmg2npy]

        # Delete the rwo files
        if self.SG_flag and self.PRES_flag:
            try:
                del_folder = os.path.join(self.simfolder, self.batchfolder, f'rwo_{case_name}')
                shutil.rmtree(del_folder)
            except:
                if self.err_stop:
                    raise ValueError(f'{case_name} SG & PRES cmgrst to npy are successful, but rwo folder not completely deleted ...')
                else:
                    print(f'{case_name} SG & PRES cmgrst to npy are successful, but rwo folder not completely deleted ...')

    def read_SG_rwo2npy(self, case_name, save=True):
        cmgrst = pycmgresults()
        cmgrst.XY2arr_interp_method = self.XY2arr_interp_method
        cmgrst.XY2arr_interp_num_x = self.XY2arr_interp_num_x
        cmgrst.XY2arr_interp_num_y = self.XY2arr_interp_num_y

        rwo_dir = os.path.join(self.simfolder, self.batchfolder, f'rwo_{case_name}')

        try:
            x_new, y_new, SG_arr = cmgrst.rwo_reader2arr(folder=rwo_dir,
                                                         sim=case_name,
                                                         prop='SG',
                                                         layer_nums=self.layer_nums,
                                                         time_query=[f'Gas Saturation_{t}-Jan-01' for t in self.time_query],
                                                         x_dir_key=self.x_dir_key, y_dir_key=self.y_dir_key)
            self.cmg2npy = SG_arr
            self.cmg2npy_x_coord = x_new
            self.cmg2npy_y_coord = y_new
            if save == True:
                np.save(os.path.join(self.npy_folder, f'{case_name}_SG.npy'), SG_arr)
                return True
            else:
                return SG_arr
            
        except:
            if self.err_stop:
                raise ValueError(f'{case_name} gas saturation has an error when reading rwo to npy ...')
            else:
                print(f'{case_name} gas saturation has an error when reading rwo to npy ...')
        
    def read_PRES_rwo2npy(self, case_name, save=True):
        cmgrst = pycmgresults()
        cmgrst.XY2arr_interp_method = self.XY2arr_interp_method
        cmgrst.XY2arr_interp_num_x = self.XY2arr_interp_num_x
        cmgrst.XY2arr_interp_num_y = self.XY2arr_interp_num_y

        rwo_dir = os.path.join(self.simfolder, self.batchfolder, f'rwo_{case_name}')

        try:
            x_new, y_new, PRES_arr = cmgrst.rwo_reader2arr(folder=rwo_dir,
                                                           sim=case_name,
                                                           prop='PRES',
                                                           layer_nums=self.layer_nums,
                                                           time_query=[f'Pressure_{t}-Jan-01' for t in self.time_query],
                                                           x_dir_key=self.x_dir_key, y_dir_key=self.y_dir_key)
            self.cmg2npy = PRES_arr
            self.cmg2npy_x_coord = x_new
            self.cmg2npy_y_coord = y_new
            if save == True:
                np.save(os.path.join(self.npy_folder, f'{case_name}_PRES.npy'), PRES_arr)
                return True
            else:
                return PRES_arr
            
        except:
            if self.err_stop:
                raise ValueError(f'{case_name} pressure has an error when reading rwo to npy ...')
            else:
                print(f'{case_name} pressure has an error when reading rwo to npy ...')
            
        
    def read_VERDSPLGEO_rwo2npy(self, case_name, save=True):
        cmgrst = pycmgresults()
        cmgrst.XY2arr_interp_method = self.XY2arr_interp_method
        cmgrst.XY2arr_interp_num_x = self.XY2arr_interp_num_x
        cmgrst.XY2arr_interp_num_y = self.XY2arr_interp_num_y

        rwo_dir = os.path.join(self.simfolder, self.batchfolder, f'rwo_{case_name}')

        ######################################################################################
        ##### No need to have this anymore ....
        # # For cases with changing injection horizon
        # if self.inj_hrzn:
        #     self.time_query = list(np.arange(self.inj_hrzn+1)+self.time_start_year)
        #     if self.yr_after_shutin_disp:
        #         for yy in self.yr_after_shutin_disp:
        #             self.time_query.append(self.inj_hrzn+self.time_start_year+yy)
        # else:
        #     print(f"Injection horizon is None, no time query for CMG result extraction ...")
        ######################################################################################

        try:

            x_new, y_new, VERDSPLGEO_arr = cmgrst.rwo_reader2arr(folder=rwo_dir,
                                                                 sim=case_name,
                                                                 prop='Vertical Displacement from Geomechanics',
                                                                 layer_nums=self.layer_nums,
                                                                 time_query=[f'Vertical Displacement from Geomechanics_{t}-Jan-01' for t in self.time_query],
                                                                 x_dir_key=self.x_dir_key, y_dir_key=self.y_dir_key)

            self.cmg2npy = VERDSPLGEO_arr
            self.cmg2npy_x_coord = x_new
            self.cmg2npy_y_coord = y_new
            
            if save == True:
                np.save(os.path.join(self.npy_folder, f"{case_name.split('.')[0]}_VERDSPLGEO.npy"), VERDSPLGEO_arr)
                return True
            else:
                return VERDSPLGEO_arr
            
        except:
            if self.err_stop:
                raise ValueError(f'{case_name} VERDSPLGEO has an error when reading rwo to npy ...')
            else:
                print(f'{case_name} VERDSPLGEO has an error when reading rwo to npy ...')
            



    
    def cmgrst2npy(self, caseid, verbose=False, rwodelete=True):
        """
        Assemble components to read from CMG sr3 results to property npy files
        Add flexibility for results parse.
        """
        
        if self.proplist in [['SG'], ['PRES'], ['SG','PRES'], ['PRES','SG']]:
            casename = f'case{caseid}'
        elif self.proplist in [['Vertical Displacement from Geomechanics'], ['VERDSPLGEO']]:
            casename = f'case{caseid}.gmch'
        else:
            if self.err_stop:
                raise ValueError('Error: the prop cannot found in cmgrst2npy function in pyCMG_Control.py file ...')
            else:
                print('Error: the prop cannot found in cmgrst2npy function in pyCMG_Control.py file ...')
            
        
        # STEP1: write cmg rwd file to simfolder
        self.wrt_rwd_report(case_name=casename, verbose=verbose)

        # STEP2: check all necessary folders available (if not, create folders)
        self.folder_sanity_check(case_name=casename)

        # STEP3: run CMG result report module (.exe): read rwd and generate rwo files
        self.run_rwd_report(case_name=casename)

        # STEP4: read rwo files to npy
        if self.proplist == ['SG']:
            self.save_done = self.read_SG_rwo2npy(case_name=casename, save=True)

        elif self.proplist == ['PRES']:
            self.save_done = self.read_PRES_rwo2npy(case_name=casename, save=True)

        elif self.proplist in [['SG','PRES'], ['PRES','SG']]:
            self.read_PRES_SG_from_rwo(case_name=casename)
            self.save_done = False

        elif self.proplist == ['Vertical Displacement from Geomechanics']:
            self.save_done = self.read_VERDSPLGEO_rwo2npy(case_name=casename, save=True)

        else:
            if self.err_stop:
                raise ValueError('Error: the prop cannot found in cmgrst2npy_v2 function in pyCMG_Control.py file ...')
            else:
                print('Error: the prop cannot found in cmgrst2npy_v2 function in pyCMG_Control.py file ...')

        # Delete unnecessary files
        if self.save_done and rwodelete:
            try:
                del_folder = os.path.join(self.simfolder, self.batchfolder, f'rwo_{casename}')
                shutil.rmtree(del_folder)
                del_rwd = os.path.join(self.simfolder, self.batchfolder, f'{casename}.rwd')
                os.remove(del_rwd)
            except:
                if self.err_stop:
                    raise ValueError(f'{casename} {self.proplist} cmgrst to npy are successful, but rwo folder not completely deleted ...')
                else:
                    print(f'{casename} {self.proplist} cmgrst to npy are successful, but rwo folder not completely deleted ...')

        


    def cmgrst2npy_v1(self, caseid, rst_type='sr3'):
        """
        Assemble components to read from CMG sr3 results to property npy files
        """
        # STEP1: write cmg rwd file to simfolder
        if rst_type == 'sr3':
            casename = f'case{caseid}'
        elif rst_type == 'gmch.sr3':
            casename = f'case{caseid}.gmch'
        else:
            if self.err_stop:
                raise ValueError('Error: the rst_type is given with a wrong value .....')
            else:
                print('Error: the rst_type is given with a wrong value .....')
        self.wrt_rwd_report(case_name=casename, verbose=False)

        # STEP2: check all necessary folders available (if not, create folders)
        self.folder_sanity_check(case_name=casename)

        # STEP3: run CMG result report module (.exe): read rwd and generate rwo files
        self.run_rwd_report(case_name=casename)

        # STEP4: read rwo files to npy
        if rst_type == 'sr3':
            self.read_PRES_SG_from_rwo(case_name=casename)
        elif rst_type == 'gmch.sr3':
            self.read_VERDSPLGEO_rwo2npy(case_name=casename)
        else:
            if self.err_stop:
                raise ValueError('Error: the rst_type is given with a wrong value .....')
            else:
                print('Error: the rst_type is given with a wrong value .....')


    def readcmg_ts_rwo_to_dict(self, path2file, line_dist):
        """
        Input
        path2file: path to the rwo file. e.g. '../data/Shell/testoutput.rwo'
        line_dist: number of lines from the line of 'FILE: xxx.sr3' to the first line of the numbers
        E.g.:   Shell case, line_dist = 6
                Case Gasification model CASE 1.sr3, line_dist = 9
        Output:
        cache:  dict format.
                Keys record the file names if multiple files extract results all together.
                Values: list of lists. Length is the number of lines of data. For each sub-list, length is the number of columns in rwo files.

        """
        with open(path2file) as file:
            lines = file.readlines()

        cache = {}
        i = 0

        while i < len(lines):
            # try:
            if lines[i].split() and lines[i].split()[0] == 'FILE:':
                # caseid = lines[i].split()[-1][0:-4]
                caseid = ' '.join(lines[i].split()[1:-1]+[''])+lines[i].split()[-1][0:-4]
                if caseid not in cache:
                    cache[caseid] = []
                dline = line_dist
                temp_list = []
                while (i+dline<len(lines)) and lines[i+dline].split() and self.isfloat(lines[i+dline].split()[0]):
                    temp_list.append([float(lines[i+dline].split()[0]), float(lines[i+dline].split()[1])])
                    dline += 1
                cache[caseid] += temp_list
                i += dline
            else:
                i += 1

        return cache
    
    def isfloat(self, num):
        try:
            float(num)
            return True
        except ValueError:
            return False

    def save_dict_to_npy(self, cache, save_folder):
        for case in cache:
            data = np.array(cache[case])
            np.save(file=os.path.join(save_folder, f'{case}.npy'), arr=data)

        return data
