'''
Author: Yunan Li
Date: 2024/02/14
Goal: history match updated files for pyCMG
'''
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import sys
# append the path of the parent directory
sys.path.append("..")
from utils.pyCMG_Model import reaction_h2, global_sa
from utils.pyCMG_Control import pycmgcontrol
from utils.pyCMG_Simulator import wrt_cmgrwd_ts_special, run_cmgrwd
        

from scipy import interpolate
from scipy.optimize import differential_evolution





class historymatch_reaction():
    def __init__(self, exp_name, expfolder):
        super().__init__()
        self.exp_name = exp_name
        self.expfolder = expfolder
        self.list_of_HMparams = ['c12','c14','c22','c52','A1','E1','A2','E2','A5','E5']
        self.list_of_constrained_params = []
        self.expH2 = np.array([])

        ##### Optimization log #####
        self.loss_func_values = []
        self.num_iteartion = 0
        self.num_simruns = 0

        ##### DE optimizer set-up #####
        print('Optimization bounds are required!')
        self.bounds = [(0,5), (0, 5), (0, 5), (0, 5), (10**6, 10**12), (0, 10**8), (10**6, 10**12), (0, 10**8), (10**6, 10**12), (0, 10**8)]
        self.strategy = 'best1bin'
        self.maxiter = 1000
        self.popsize = 15
        self.tol = 0.1
        self.seed = None
        self.disp = False
        self.polish = True
        self.init = 'latinhypercube'
        self.updating = 'immediate'
        self.vectorized = False
        self.constraints = ()
        self.x0 = None  # ENTER THIS AS LIST or ARRAY

        ##### Create folders if necessary
        # Create the experiment folder
        if not os.path.exists(self.expfolder):
            try:
                os.makedirs(self.expfolder)
            except:
                pass
        # Crease simulation folder
        if not os.path.exists(os.path.join(self.expfolder, 'simfiles')):
            try:
                os.makedirs(os.path.join(self.expfolder, 'simfiles'))
            except:
                pass
        # Create figures folder
        if not os.path.exists(os.path.join(self.expfolder, 'figures')):
            try:
                os.makedirs(os.path.join(self.expfolder, 'figures'))
            except:
                pass


    def calc_distance(self, t_hm, y_hm, y_pred, dist_type='MSE'):
        """
        t_hm and y_hm need to have the same length.
        y_pred needs to be an array.
        """

        t_min, t_max = np.min(t_hm), np.max(t_hm)
        t_uni = np.linspace(t_min, t_max, num=y_pred.shape[0])
        f_hm = interpolate.interp1d(t_hm, y_hm)
        y_hm_uni = f_hm(t_uni)

        if dist_type.lower() == 'mse':
            mse = np.square(np.subtract(y_hm_uni,y_pred)).mean()
            return mse
        elif dist_type.lower() == 'rmse':
            rmse = np.sqrt(np.square(np.subtract(y_hm_uni,y_pred)).mean())
            return rmse
        else:
            print('This dist_type is not available in calc_distance function of historymatch class ...')


    def CMG_exec_wrtdat_to_rstnpy(self, x):

        df_params = self.map_x_array2df(x)
        self.run_name = f'Iteartion{self.num_iteartion}_Simrun{self.num_simruns}.dat'
        # Write CMG dat file (model)
        rxnh2 = reaction_h2()
        rxnh2.write_dat_file(folder_path=os.path.join(self.expfolder, 'simfiles'),
                             dat_file_name=self.run_name,
                             df_params=df_params)
        # Run CMG simulation
        pycmg_ctrl = pycmgcontrol(exp_name=self.exp_name, simfolder=os.path.join(self.expfolder, 'simfiles'))
        pycmg_ctrl.cmg_version = 'ese-win32-v2022.30'
        pycmg_ctrl.run_stars_simulation(case_name_suffix=self.run_name)

        # Create folder and inputs for rwd/rwo files step
        read_prop = 'MOLEFRAC H2 G OUTLET'
        rwo_files_folder = 'rwo_data'
        if not os.path.exists(os.path.join(self.expfolder, 'simfiles', rwo_files_folder)):
            try:
                os.makedirs(os.path.join(self.expfolder, 'simfiles', rwo_files_folder))
            except:
                pass
        # Write rwd file to extract results
        wrt_cmgrwd_ts_special(sim_sr3=f'Iteartion{self.num_iteartion}_Simrun{self.num_simruns}',
                            ext_rwo=None,
                            path2rwd=os.path.join(self.expfolder, 'simfiles'),
                            create_rwo_folder=True,
                            rwo_folder=rwo_files_folder,
                            prop=read_prop,
                            precis=4,
                            cmg_version='ese-win32-v2022.30',
                            verbose=True)
        
        # Run CMG rwd to generate rwo files
        pycmg_ctrl.run_rwd_report(case_name=f'Iteartion{self.num_iteartion}_Simrun{self.num_simruns}')
        # Extract results from rwo files
        cache = pycmg_ctrl.readcmg_ts_rwo_to_dict(path2file=os.path.join(self.expfolder, 'simfiles', rwo_files_folder, f'Iteartion{self.num_iteartion}_Simrun{self.num_simruns}_{read_prop}.rwo'),
                                                  line_dist=9)
        
        # Save to rst_npy folder
        self.npy_folder = os.path.join(self.expfolder, 'simfiles', 'rst_npy')
        if not os.path.exists(self.npy_folder):
            try:
                os.makedirs(self.npy_folder)
            except:
                pass
        # npy_data is the last npy if multiple files in rwo or in cache
        npy_data = pycmg_ctrl.save_dict_to_npy(cache=cache,
                                               save_folder=self.npy_folder)
        return npy_data
        


    def loss_function(self, x):
        """
        Optimization objective function defined here! 
        Input x: parameter vector. It has to agree with the self.bounds definition. It maps to a df for CMG runs.
        Return loss: scalar value. 

        Attention:
        1. Add details information in this function while DE calls it for solutions.
        """
        loss = 0
        # Get input for CMG
        # df_params = self.map_x_array2df(x)


        # Plot experimet
        colors = ['b', 'r', 'm', 'c', 'g', 'y']
        plt.figure()

        # Compute loss
        try:
            rstnpy = self.CMG_exec_wrtdat_to_rstnpy(x)
            loss += self.calc_distance(t_hm=rstnpy[:,0],
                                        y_hm=rstnpy[:,1],
                                        y_pred=self.expH2[:,1],
                                        dist_type='MSE')
            
            self.num_simruns += 1

            plt.plot(self.expH2[:,0], self.expH2[:,1], colors[0], label='exp')
            plt.plot(rstnpy[:,0], rstnpy[:,1], colors[1], label='sim')
            
        except:
            loss += 1e5


        self.num_iteartion += 1

        plt.xlabel('Time', fontsize=16)
        plt.ylabel('MOLEFRAC H2 G OUTLET', fontsize=16)
        plt.title(f'Iteartion{self.num_iteartion}_Simrun{self.num_simruns}', fontsize=16)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.legend()
        plt.savefig(os.path.join(self.expfolder, 'figures', f'Iteartion{self.num_iteartion}_Simrun{self.num_simruns}.png'), bbox_inches='tight', dpi=300)
        plt.close()

        # Log reaction
        self.print_reaction_from_params(x, file_name=self.log_file)
        # Log loss value
        fid = open(self.log_file, 'a+')
        print(f'Loss value at {self.num_iteartion} iteartion of {self.num_simruns} simulation: {loss}',file=fid)
        print('='*100, file=fid)
        fid.close()
        self.loss_func_values.append(loss)

        return loss


    def optimize_loss_DE(self):

        ##### User defined initial x0 or not ??? #####
        if self.x0 is None:
            pass
        else:
            self.init = np.expand_dims(self.x0, 0) + np.random.randn(self.popsize, len(self.x0))

        ##### Create log file
        self.log_file = os.path.join(self.expfolder, f'{self.exp_name}_logfile.txt')
        fid = open(self.log_file,'w+')
        print(f'Initializing history match {self.exp_name} ...', file=fid)
        print('Optimization set-up details attached', file=fid)
        print(f'strategy: {self.strategy}', file=fid)
        print(f'maxiter: {self.maxiter}', file=fid)
        print(f'popsize: {self.popsize}', file=fid)
        print(f'tol: {self.tol}', file=fid)
        print(f'seed: {self.seed}', file=fid)
        print(f'disp: {self.disp}', file=fid)
        print(f'polish: {self.polish}', file=fid)
        print(f'init: {self.init}', file=fid)
        print(f'updating: {self.updating}', file=fid)
        print(f'vectorized: {self.vectorized}', file=fid)
        print('='*66, file=fid)
        fid.close()

        sol = differential_evolution(func = self.loss_function, 
                                     bounds = self.bounds, 
                                     strategy = self.strategy,
                                     maxiter = self.maxiter,
                                     popsize = self.popsize, 
                                     tol = self.tol,
                                     seed = self.seed,
                                     disp = self.disp,
                                     polish = self.polish, 
                                     init = self.init,
                                     updating = self.updating,
                                     constraints = self.constraints
                                    #  vectorized = self.vectorized
                                     )
        
        # Write details in log file
        fid = open(self.log_file, 'a+')
        print('+'*100, file=fid)
        print(sol.message, file=fid)
        print('Final optimized results:', file=fid)
        fid.close()
        self.print_reaction_from_params(sol.x, file_name=self.log_file)

        # Print the final result
        fid = open(self.log_file, 'a+')
        print('The minimum functional loss value is '+str(np.min(self.loss_func_values))+' at the ' +str(1+self.loss_func_values.index(min(self.loss_func_values))) +'th optimization case', file=fid)
        # print('The minimum stars (cmg) loss value is '+str(np.min(self.loss_values_stars))+' at the ' +str(1+self.loss_values_stars.index(min(self.loss_values_stars))) +'th optimization case', file=fid)
        print('+'*100, file=fid)
        fid.close()

        # Save the loss values in csv
        dfloss = pd.DataFrame()
        dfloss['Loss'] = self.loss_func_values
        dfloss.to_csv(os.path.join(self.expfolder, 'Loss_values.csv'), index=False)

        # Print convergence plot
        plt.figure()
        plt.semilogy(self.loss_func_values)
        plt.xlabel('Number of iterations', fontsize=16)
        plt.ylabel('Loss value', fontsize=16)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.title('Optimization Convergence Plot', fontsize=16)
        plt.savefig(os.path.join(self.expfolder, 'convergence_plot_log.png'), bbox_inches='tight', dpi=300)
        plt.close()

        # Print convergence plot
        plt.figure()
        plt.plot(self.loss_func_values)
        plt.xlabel('Number of iterations', fontsize=16)
        plt.ylabel('Loss value', fontsize=16)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.title('Optimization Convergence Plot', fontsize=16)
        plt.savefig(os.path.join(self.expfolder, 'convergence_plot.png'), bbox_inches='tight', dpi=300)
        plt.close()

        # plt.figure()
        # plt.semilogy(self.loss_values_stars)
        # plt.xlabel('Number of stars evaluations')
        # plt.ylabel('Loss value')
        # plt.title('Optimization Convergence Plot')
        # plt.savefig(os.path.join(self.results_dir, 'convergence_plot_stars.png'))




    def map_x_array2df(self, x):
        df_params = {}
        for ii,pp in enumerate(self.list_of_HMparams):
            df_params[pp] = x[ii]
        
        ##### Hard code the constraints #####
        # for jj, cp in enumerate(self.list_of_constrained_params):
        #     df_params[cp] = function_of_constraints
        
        df_params['c13'] = (0.0159+df_params['c12']*0.01802-df_params['c14']*0.002016)/0.02801
        df_params['c23'] = (0.0159+df_params['c22']*0.04401)/0.02801
        df_params['c53'] = (0.0159+df_params['c52']*0.002016)/0.01604
        return df_params



    def print_reaction_from_params(self, x, file_name=None):
        df_params = self.map_x_array2df(x)

        if file_name is not None:
            fid = open(file_name, 'a+')
        else:
            fid = None

        print(f"Reaction at {self.num_iteartion} iteartion of {self.num_simruns} simulation: \n",file=fid)

        print(f"Coke2 + {df_params['c12']} H2O = {df_params['c13']} CO + {df_params['c14']} H2", file=fid)
        print(f"A={df_params['A1']}, E={df_params['E1']}, H=0 \n", file=fid)

        print(f"Coke2 + {df_params['c22']} CO2 = {df_params['c23']} CO", file=fid)
        print(f"A={df_params['A2']}, E={df_params['E2']}, H=0 \n", file=fid)

        print(f"Coke2 + {df_params['c52']} H2 = {df_params['c53']} CH4", file=fid)
        print(f"A={df_params['A5']}, E={df_params['E5']}, H=0 \n", file=fid)

        if file_name is not None:
            fid.close()

