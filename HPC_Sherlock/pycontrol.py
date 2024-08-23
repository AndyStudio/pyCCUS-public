import os
# Users need to change the dat name accordingly ...
dat_name = 'case0.dat'
#####
pwd_CMG = "/home/groups/sh_s-dss/share/sdss/software/x86_64_arch/CMG/2023.101/gem/2023.10/linux_x64/exe/gm202310.exe"
pwd = os.getcwd()
t = 'cd ' + pwd + ' &' + str(pwd_CMG) + '  -f ' + dat_name
os.system(t)
