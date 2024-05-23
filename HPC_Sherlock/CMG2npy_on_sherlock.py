def CMG2npy(case_idx, props, lyr_query, time_query, prop_dict):
    simID = f"case{case_idx}"
    rwo = f"rwo_case{case_idx}"
    flag = True
    ##### func write to pwd folder #####
    try:
        wrt_cmgrwd(sim_sr3=simID,
                ext_rwo=None,
                path2simfolder=None,
                create_rwo_folder=True,
                rwo_folder=rwo,
                proplist=props,
                layer_num=lyr_query,
                time_step='*ALL-TIMES',
                layer_type = '*XYZLAYER',
                precis=4)
    except:
        print(f'Case{case_idx} write CMG rwd file failed .....')
        flag = False

    isExist = os.path.exists(rwo)
    if not isExist:
        os.makedirs(rwo)
    try:
        run_rwd(case_idx)
    except:
        print(f'Case{case_idx} run CMG rwd file failed .....')
        flag = False

    npy = 'rst_npy'
    if not os.path.exists(npy):
        os.makedirs(npy)

    try:
        for pp in props:
            arr = rwo_reader2arr(folder=rwo,
                         sim=simID,
                         prop=pp,
                         layer_nums=lyr_query,
                         time_query=[f'{prop_dict[pp]}_{t}-Jan-01' for t in time_query])

            np.save(os.path.join(npy, f'{simID}_{pp}.npy'), arr)
            print(f'Case{case_idx} results written in npy done .....')

    except:
        print(f'Case{idx} results to npy failed .....')
        flag = False

    # Delete all rwo and rwd files
    if flag == True:
        try:
            os.remove(f'{simID}.rwd')
            shutil.rmtree(rwo, ignore_errors=False)
            #print(f'Case{idx} completed with rwo/rwd files deleted!')
        except:
            print(f'Case{idx} rwd and rwo files deleted failed .....')

    # return print(f'Case{case_idx} results written in npy done .....')



def rwo_reader2arr(self, folder, sim, prop, layer_nums, time_query, x_dir_key='X', y_dir_key='Y'):
    """
    Goal: combine rwo_reader2df and rwodf2arr in a 4d (n,m,z,t) array for output.
    Inputs: 1. folder: path to the sim run folder, example="../data/EPA_baseline/"
            2. sim: simulation run name, example="EPA_baseline_dev2East_debug"
            3. prop: CMG keyword for property 'SG', 'PRES', 'VERDSPLGEO', etc.
            4. layer_nums: list or array in 1d
            5. query: in format of [list] or arr for years. example=['Gas Saturation_2027-Jan-01', ..., 'Gas Saturation_2100-Jan-01']
    """

    df0 = self.read_rwo2csv(f'{folder}{sim}_{prop}_layer{layer_nums[0]}.rwo')
    arr0 = self.rwodf2arr(df=df0, query=time_query[0], x_dir_key='X', y_dir_key='Y')
    n,m = arr0.shape
    z = len(layer_nums)
    t = len(time_query)

    rst_arr = np.zeros((n,m,z,t))

    for il,ll in enumerate(layer_nums):
        globals()[f'df{prop}_lyr{ll}'] = self.read_rwo2csv(f'{folder}{sim}_{prop}_layer{ll}.rwo')
        for it, tt in enumerate(time_query):
            globals()[f'arr{prop}_lyr{ll}_{tt}'] = self.rwodf2arr(df=globals()[f'df{prop}_lyr{ll}'], query=tt, x_dir_key=x_dir_key, y_dir_key=y_dir_key)
            rst_arr[:,:,il,it] = globals()[f'arr{prop}_lyr{ll}_{tt}']

    return rst_arr


def wrt_cmgrwd(sim_sr3=simID,
                ext_rwo=None,
                path2simfolder=None,
                create_rwo_folder=True,
                rwo_folder=rwo,
                proplist=props,
                layer_num=lyr_query,
                time_step='*ALL-TIMES',
                layer_type = '*XYZLAYER',
                precis=4):
    """
    The function is in a different rpo. Will be updated soon ...
    """

def run_rwd(case_idx):
    """
    The function is in a different rpo. Will be updated soon ...
    """



import numpy as np
import shutil
import os

idx = 1
proplist = ['VERDSPLGEO']
lyr_query = np.arange(1,76)+40
time_query = [2023+i for i in range(0,120)]

prop_dict = {}
prop_dict['SG'] = 'Gas Saturation'
prop_dict['PRES'] = 'Pressure'
prop_dict['VERDSPLGEO'] = 'Vertical Displacement from Geomechanics'


CMG2npy(case_idx=idx, props=proplist, lyr_query=lyr_query, time_query=time_query, prop_dict=prop_dict)
