import os
import pandas as pd
import glob
import numpy as np



class pysanitycheck():
    def __init__(self):
        super().__init__()


    def rwo_sanity_check(self, exp_folder_path, rwo_folderID, num_files):
        """
        Goal: count the number of rwo files in the folders to screen cases with unexpected errors.
        Return: 2 lists of ints, where are ids for good folders and folders with problems.
        Inputs:
        exp_folder_path: the path to the expected folder that is parent folder of all the rwo folders
                        (e.g. well opt exp folder)
        rwo_folderID: list of ints. Represent the ids for the rwo folders to check. e.g.: [1,2,3,...,120]
        num_files: int single value. This is the expected number of files under the rwo folders we check.
        """
        yes_cnt, no_cnt = [],[]
        for i in rwo_folderID:
            dir2path = os.path.join(exp_folder_path,f'rwo_{i}')
            filenames = os.listdir(dir2path)
            if len(filenames) == num_files:
                yes_cnt.append(i)
            else:
                no_cnt.append(i)
        
        if len(no_cnt) == 0:
            print('Files in rwo folder check: pass!')
        else:
        
            print(f'Files in rwo folder check: {len(yes_cnt)} of cases meet the expectation and {len(no_cnt)} of cases need further actions .....')
        
        return yes_cnt,no_cnt
        
    def cmgsim_sanity_check(self, sim_folder_path, sim_caseid, files_ext, checkall=False, sim_basename=None):
        """
        Logic: check the number of files in the folder (look at the file extensions to check) 
        Goal: in case some wired situation that some cases are not generated successfully.
        """
        
        yes_cnt, no_cnt = [],[]
        if sim_basename is None:
            sim_basename = 'case'
            
        ext_yn_cnt = 0
        
        for ext in files_ext:
            if len(glob.glob(os.path.join(sim_folder_path,f'*.{ext}'))) % len(sim_caseid) ==0:
                # In case to tell apart extensions like .gmch.sr3 and .sr3
                ext_yn_cnt += 1
        
        if ext_yn_cnt == len(files_ext):
            print('CMG files check: pass!')
        if ext_yn_cnt != len(files_ext) or checkall==True:
            
            for ii in sim_caseid:
                if len(glob.glob(os.path.join(sim_folder_path,f'{sim_basename}{ii}.*'))) == len(files_ext):
                    yes_cnt.append(ii)
                else:
                    no_cnt.append(ii)
            if len(no_cnt)==0:
                print('CMG files check: pass! (Extra files exist in the folder, but got enough sim files to proceed)')
            else:
                print(f'CMG files check: {len(yes_cnt)} of sim cases have correct number of files to proceed, and {len(no_cnt)} of sim cases need to check with further details .....')
        elif checkall==False:
            pass
        
        else:
            print('Unexpected error in the cmgsim_sanity_check function .....')
            
        return yes_cnt, no_cnt
        

                
    def rstnpy_sanity_check(self, rst_folder_path, sim_caseid, rst_type, sim_basename=None):
        """
        Logic: check the npy files in the folder for all cases.
        Look at the extensions and the file names.
        """
        yes_cnt, no_cnt = [],[]
        if sim_basename is None:
            sim_basename = 'case'
            
        type_yn_cnt = 0
        
        for tp in rst_type:
            if len(glob.glob(os.path.join(rst_folder_path,f'*_{tp}.npy'))) == len(sim_caseid):
                type_yn_cnt += 1
        
        if type_yn_cnt == len(rst_type):
            print('Result (.npy) files check: pass!')
        else:
            Flag = True
            for ii in sim_caseid:
                if len(glob.glob(os.path.join(rst_folder_path,f'case{ii}_*.npy'))) == len(rst_type):
                    yes_cnt.append(ii)
                else:
                    print(f'Result (.npy) files check: case{ii} files not complete .....')
                    no_cnt.append(ii)
                    Flag = False
            if Flag:
                print('Result (.npy) files check: pass! (Extra files found, but good to proceed)')
        
        return yes_cnt, no_cnt


    def check_slurm_out_termination_status(self, folder_path, file_name):

        """
        Logic: read the slrum .out files => heck the bottom few lines => End of simulaiton status
        Return: 0 (normal termination) or 1 (abnormal termination)
        """
        path2file = os.path.join(folder_path, file_name)

        with open(path2file) as file:
            lines = file.readlines()

        nn = len(lines)-1

        while nn >= 0:
            ll = lines[nn]
            if ll.split(':')[0].split() == ['End', 'of', 'Simulation']:
                if ll.split(':')[1].split() == ['Abnormal', 'Termination']:
                    return 1
                elif ll.split(':')[1].split() == ['Normal', 'Termination']:
                    return 0
                else:
                    print(f"{file_name} End of Simulation status in slurm out file is incorrect ...")
            else:
                nn -= 1

        print(f"{file_name} cannot find the End of Simulation status in slurm out file ...")
        return 55555
    
    def check_slurm_out_termination_status_in_batch(self, folder_path, file_name_list, verbose=False):
        """
        Return a list of 0 and 1 as term. status for sanity check.
        file_name should be a list of items like this: job_1.out.
        """
        # df = pd.DataFrame({})
        term_status = []
        for fn in file_name_list:
            status_temp = self.check_slurm_out_termination_status(folder_path, fn)
            term_status.append(status_temp)

        if verbose:
            print(f"{np.round(np.sum(term_status)/len(file_name_list)*100, 2)}% of the cases are successful (from CMG simulation) ...")
        
        return term_status


