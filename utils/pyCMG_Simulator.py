'''
Author: Yunan Li
Date: 2023/01/15
Goal: this file contains necessary functions to generate rwd file based on simulation result (sr3 files), 
      and then execute it in windows local or Linux (sherlock) to actually run rwd file in CMG Results Report 
      to generate all necessary rwo files.
    
    The results extracted from CMG to Python are based on the rwo files.
'''


import os

def wrt_cmgrwd_ts_special(sim_sr3='CCS_GEM',
                        ext_rwo=None,
                        path2rwd=None,
                        create_rwo_folder=True,
                        rwo_folder='rwo_data',
                        prop='MOLEFRAC H2 G OUTLET',
                        precis=4,
                        cmg_version='ese-win32-v2022.30',
                        verbose=False):
    """
    Template for SPECIAL properties

    *FILES 	 'Gasification model CASE 1.sr3' 
    *TABLE-FOR  
    *COLUMN-FOR  *SPECIALS  'MOLEFRAC H2 G OUTLET'
    *TABLE-END 
    """
    
    # Define title lines for a rwd file
    line1 = f"*FILES \t '{sim_sr3}.sr3' \n"
    line2 = f"*PRECISION \t {precis} \n"

    # Define names for the exported rwo files
    if ext_rwo is None:
        ext_file_name = sim_sr3
    else:
        ext_file_name = ext_rwo
        
    # Define the path to write rwd file. This should be in the same folder with simulation files
    if path2rwd is None:
        rwdfile = open(f'{ext_file_name}.rwd','w')
    else:
        rwdfile = open(os.path.join(path2rwd, f'{ext_file_name}.rwd'),'w')
    rwdfile.write(line1)
    rwdfile.write(line2)

    # Separate rwo files with simulation files in a new folder
    if create_rwo_folder:
        if cmg_version == 'ese-win32-v2022.30':
            ext_file_name = f"{rwo_folder}\{ext_file_name}"
        elif cmg_version == 'ese-ts1win-v2023.20':
            ext_file_name = f"{rwo_folder}\{ext_file_name}"
        elif cmg_version == 'stf-sherlock-v2020.10':
            ext_file_name = f"{rwo_folder}/{ext_file_name}"
        else:
            raise ValueError(f'The CMG version {cmg_version} is not implemented in wrt_cmgrwd_grids function [utils.pyCMG_Simulator]')
    
    extline1 = f"*OUTPUT \t '{ext_file_name}_{prop}.rwo' \n"
    rwdfile.write(extline1)
    rwdfile.write("*TABLE-FOR  \n")
    extline2 = f"*COLUMN-FOR \t  *SPECIALS  \t '{prop}' \n"
    rwdfile.write(extline2)
    rwdfile.write("*TABLE-END  \n")
    rwdfile.close()

    if verbose == True:
        print(f"Write rwd file done from simulation: {sim_sr3}; saved to {rwo_folder}/ folder.")


def wrt_cmgrwd_grids(sim_sr3='CCS_GEM',
                    ext_rwo=None,
                    path2rwd=None,
                    create_rwo_folder=True,
                    rwo_folder='rwo_data',
                    proplist=['SG','PRES'],
                    layer_num=[1],
                    time_step='*ALL-TIMES',
                    layer_type = '*XYZLAYER',
                    precis=4,
                    cmg_version='ese-win32-v2022.30',
                    verbose=False):
    
    '''
    Note:   1. sim_sr3 only takes the name of the CMG experiment, and please do not add '.sr3' in its string.
            2. ext_rwo defines the name for exported rwo files. Default is the simulation name.
            3. path2simfolder defines the path to a folder that collects all rwo files. (This does not mix rwo files with simulation results.)
            4. create_rwo_folder (True/False) defines if we need to create a separate folder for all rwo files. (The folder name is {sim_sr3}_rwo_data\)
            5. proplist is a list of property keywords in CMG to extract grid based results.
            6. layer_num is the number of layer in CMG. It could be a list or np.arange.

    '''
    # Define title lines for a rwd file
    line1 = f"*FILES \t '{sim_sr3}.sr3' \n"
    line2 = f"*PRECISION \t {precis} \n"

    # Define names for the exported rwo files
    if ext_rwo is None:
        ext_file_name = sim_sr3
    else:
        ext_file_name = ext_rwo
        
    # Define the path to write rwd file. This should be in the same folder with simulation files
    if path2rwd is None:
        rwdfile = open(f'{ext_file_name}.rwd','w')
    else:
        rwdfile = open(os.path.join(path2rwd, f'{ext_file_name}.rwd'),'w')
    rwdfile.write(line1)
    rwdfile.write(line2)

    # Separate rwo files with simulation files in a new folder
    if create_rwo_folder:
        if cmg_version == 'ese-win32-v2022.30':
            ext_file_name = f"{rwo_folder}\{ext_file_name}"
        elif cmg_version == 'ese-ts1win-v2023.20':
            ext_file_name = f"{rwo_folder}\{ext_file_name}"
        elif cmg_version == 'stf-sherlock-v2020.10':
            ext_file_name = f"{rwo_folder}/{ext_file_name}"
        else:
            raise ValueError(f'The CMG version {cmg_version} is not implemented in wrt_cmgrwd_grids function [utils.pyCMG_Simulator]')
    
    for prop in proplist:
        for layer in layer_num:
            extline1 = f"*OUTPUT \t '{ext_file_name}_{prop}_layer{layer}.rwo' \n"
            extline2 = f"*PROPERTY-FOR \t '{prop}' \t {time_step} \t {layer_type} \t {layer} \n"
            
            rwdfile.write(extline1)
            rwdfile.write(extline2)
            
    rwdfile.close()
    if verbose == True:
        print(f"Write rwd file done from simulation: {sim_sr3}; saved to {rwo_folder}/ folder.")
    # return print(f"Write rwd file done from simulation: {sim_sr3}; saved to {rwo_folder}/ folder.")
    
def run_cmgrwd(rwdfile,env='winlocal'):
    
    if env=='winlocal':
        path2CMG = '"C:\Program Files (x86)\CMG\RESULTS\2021.10\Win_x64\exe\Report"'
    elif env=='sherlock':
        path2CMG = '"/home/groups/s-ees/share/cees/software/x86_64_arch/CMG/2022.101/br/2022.10/linux_x64/exe/report.exe"'
    else:
        raise TypeError("This type of env in run_cmgrwd function hsa not been implemented yet .....")
            
    exec_line = f"cd {os.getcwd()} &{path2CMG} -f {rwdfile}"
    os.system(exec_line)
    
    return print(f"Successfully execute {rwdfile} for required rwo files.")
    
        


