'''
Author: Yunan Li
Date: 2023/05/15
Goal: this file contains necessary functions to generate rwd file based on simulation result (sr3 files), 
      and then execute it in windows local or Linux (sherlock) to actually run rwd file in CMG Results Report 
      to generate all necessary rwo files.
    
    The results extracted from CMG to Python are based on the rwo files.
'''

import os

class pysherlock():
    def __init__(self):
        super().__init__()
    
    def write_pyCTRLfile(self, folder_path, caseid):
        """
        Goal: write a single pycontrol file in the assigned folder for simulation on Sherlock.
        Run CMG dat files

        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        pyCTRL_name = f'pycontrol_{caseid}.py'
        file_path = os.path.join(folder_path, pyCTRL_name)
        fileID = open(file_path, 'w+')
        print('import os',file=fileID)
        print(f"dat_name = '../simfiles/case{caseid}.dat'",file=fileID)
#        print("""pwd_CMG = "/home/groups/s-ees/share/cees/software/x86_64_arch/CMG/2020.109/gem/2020.11/linux_x64/exe/gm202011.exe"  """,file=fileID
#        The new CMG exe path in sherlock is updated (20230921)
        print("""pwd_CMG = "/home/groups/sh_s-dss/share/sdss/software/x86_64_arch/CMG/2023.101/gem/2023.10/linux_x64/exe/gm202310.exe"  """,file=fileID)
        print('pwd = os.getcwd()',file=fileID)
        print("t = 'cd ' + pwd + ' &' + str(pwd_CMG) + '  -f ' + dat_name ",file=fileID)
        print('os.system(t)',file=fileID)
        fileID.close()

    def write_pyCTRLfile_rwd(self, folder_path, caseid):

        """
        Goal: write a single pycontrol file in the assigned folder for result extraction on Sherlock.
        Run CMG rwd files to generate rwo files

        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        pyCTRL_name = f'pycontrol_rwd_{caseid}.py'
        file_path = os.path.join(folder_path, pyCTRL_name)
        fileID = open(file_path, 'w+')
        print('import os',file=fileID)
        print(f"dat_name = '../simfiles/case{caseid}.rwd'",file=fileID)
#        print("""pwd_CMG = "/home/groups/s-ees/share/cees/software/x86_64_arch/CMG/2020.109/gem/2020.11/linux_x64/exe/gm202011.exe"  """,file=fileID
#        The new CMG exe path in sherlock is updated (20230921)
        print("""pwd_CMG = "/home/groups/sh_s-dss/share/sdss/software/x86_64_arch/CMG/2023.101/br/2023.10/linux_x64/exe/report.exe"  """,file=fileID)
        print('pwd = os.getcwd()',file=fileID)
        print("t = 'cd ' + pwd + ' &' + str(pwd_CMG) + '  -f ' + dat_name ",file=fileID)
        print('os.system(t)',file=fileID)
        fileID.close()