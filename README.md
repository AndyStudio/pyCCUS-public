# pyCCUS public
This repository is developed/designed by Yunan Li from SUETRI-A research group at Stanford University. 


### Note to users
- The public version of pyCCUS reserves functionalities due to the requirements of collaborators for multiple projects as our research moves forward.
- We demonstrate our capabilities according to applications on Geological Carbon Storage (GCS) assets.
- Please feel free to reach out if you need additional support, and we are happy to help :)
- Contact: yunanli@stanford.edu / ylistanford@gmail.com



# Selected outcomes for demonstration

## Injector design optimization
**Goal**: find the best injector trajectory to
- Minimize CO2 plume size
- Minimize pressure change in subsurface
- Minimize pressure responses within the fault cautious zone in subsurface

### Fate of all cases measured by chosen evaluation metrics.

![GCS-wellopt-animation-clip](https://github.com/AndyStudio/pyCCUS-public/assets/39730681/123c9dd9-deb6-4e89-b2d3-1765b5d229ba)


**Notation**
- Color of bubbles: maximum pressure build-up within the fault cautious zone.
- Size of bubbles: pressure footprint size, sqkm
- Horizontal axis: maximum pressure increase within the storage formation.
- Vertical axis: CO2 plume size, sqkm.


## AI-assisted GCS asset monitoring using InSAR

### Monitoring saturation plume growth throughout GCS projects.

![SG-with-yr](https://github.com/AndyStudio/pyCCUS-public/assets/39730681/32f59e90-eb3c-494d-a638-0347303c87d5)


**Notation**
- InSAR images represent the measured land surface movements (in unit of mm).
- Ground truth represents the computational results from numerical simulation.
- AI prediction is the outcome of our pre-trained image-to-image model given InSAR images for a field case.
- Color of black represents the area within the plume of saturaiton.
<!-- - Threshold is defined to be irreducible gas saturation to deliniate the plume. -->


### Pressure change (pressure build-up for GCS assets) surveillance through InSAR observations.

![PRES-with-yr](https://github.com/AndyStudio/pyCCUS-public/assets/39730681/0a351ef0-624d-4c9b-b966-466453d944aa)


**Notation**
- InSAR images represent the measured land surface movements (in unit of mm).
- Ground truth represents the computational results from numerical simulation.
- AI prediction is the outcome of our pre-trained image-to-image model given InSAR images for a field case.
- Color of black represents area within the footprints of pressure change.
<!-- - Threshold is defined to be 1 MPa as an example to describe pressure footprints. -->



# pyCCUS overview

- Automate large number of simulations needed to analyze Geological Carbon Storage (GCS) outcomes. 
- The post-processing and analysis component of this toolbox computes evaluation metrics and outcomes dynamically from numerical simulation results. 
- Support the CUSP project for CCUS (Carbon Capture, Utilization, and Storage). 
- This toolbox interacts with the commercial software CMG so that the numerical model is parsed by CMG for computations. 

**The overview of this workflow with essential components is noted.**

![Fig4](https://github.com/AndyStudio/pyCCUS-public/assets/39730681/3cb6de68-d3f6-47e4-aaf4-238e030d4ad9)



## Structure and highlights
- Start from the numerical simulation model.
- [Optional] Parameter space inputs (parameters and boundaries) from users as an **optional** choice if the goal is to drive pyCCUS to create many reservoir simulation models.
- Automatically writes a number of simulation model input files accordingly. 
- Compatible with different operating systems, such as Windows (e.g.  your local machine or workstation), Linux, multiple HPC resources (e.g. Stanford Sherlock, Shell HPC, etc.).
- Scheduler submits all jobs prioritizing parallel over sequential computations depending on the computational resources allocated.
- The “sanity check” component ensures the success of the simulation and filters out cases that do not satisfy our settings.
- A novel module to generate deviated injector trajectories for optimization.
- Optimization module
    - Forward optimization (e.g. computationally expensive model with enough computational resources for parallel computations)
    - Looped optimization (e.g. history match using a relatively simple model)
        - Example: chemical kinetics history match with experimental measurements (Li et al., 2023)
- AI module 
    - ML algorithms for regression and classification tasks, etc.
    - DL models for feature extraction, image-to-image predictions, etc.


## What you could do with pyCCUS?
- GCS field case design and strategies optimization.
- GCS post-processing and analysis.
    - 2D/3D CO2 plume delineation
    - Pressure footprint characterization
    - CO2 migration distances with uncertainties *[essential information for induced seismicity assessments (Kohli et al., 2023)]*
    - And so on ...
- Uncertainty quantification.
- Global sensitivity analysis.
- History match to calibrate numerical model and reduce simulation uncertainties.


![InSAR-HM](https://github.com/AndyStudio/pyCCUS-public/assets/39730681/3950f821-9231-40ea-845d-79b29aeebcd3)


## pyCCUS structure
``` bash
|____cmg_models
| |____SPR_CCS_case130_cartesian.dat
| |____wrtcmgdat_h2_rxns.py
| |____wrtcmgdat_SPR_CCS_field6x6.py
| |____SPR_CCS_simplified.dat
| |____SPR_CCS_case130.dat
|____.DS_Store
|____LICENSE
|____environment.yml
|____2_AE_dimension_reduction
| |____AE_DGSA.ipynb
| |____AE_rst_viz.ipynb
| |____AE_disp_SA2.py
|____4_HM_InSAR
| |____CMG_cartesian_grids.ipynb
|______init__.py
|____utils
| |____GSLIB_Petrel_remain_problems.ipynb
| |____rwo2csv.ipynb
| |____pySherlock.py
| |____history_match.py
| |____pyCMG_Results.py
| |____SPR_data_visualization.ipynb
| |____pyCMG_Control.py
| |______init__.py
| |____kh_plot_for_Raji.py
| |____wrt_cmgrwd_exe.py
| |____pyCMG_Model.py
| |____read_gslib.ipynb
| |____CCS_plume_from_CMG.ipynb
| |____pyCMG_Visualization.py
| |____read_petrel.py
| |____CCS_plume_from_CMG-WRM2023.ipynb
| |____pySanity_Check.py
| |____pyCMG_Simulator.py
|____2_GlobalSA_InSAR
| |____GlobalSA_exp2new_3D_fullmat.ipynb
| |____GlobalSA_all_in_npy_orgdata.ipynb
| |____.DS_Store
| |____RS_syntheticInSAR.ipynb
| |____GlobalSA_generate_datfiles.ipynb
|____pyCMG_copyright_v1.docx
|____CMG
| |____utils
|____README.md
|____1_SPEJ_injector_opt
| |____SPEWRM_wellopt_UQ.ipynb
| |____SPEWRM_wellopt_exp3_allcases.ipynb
| |____WellOpt_trajectory_viz.ipynb
| |____WellOpt_npy2csv_UQ_inj.ipynb
| |____RS_verdispgeo_rst2npy.ipynb
| |____WellOpt_npy2csv_UQ_rock.ipynb
| |____kphi_realizations.ipynb
| |____CCS_injection_schemes.ipynb
| |____SanityCheck_outfiles.ipynb
| |____WellOpt_traj_petrophysics.ipynb
| |____SPEJ_wellopt_exp3.ipynb
| |____SPEJ_wellopt_UQ.ipynb
| |____SPEJ_well_optimization_rst.ipynb
| |____WellOpt_npy2csv_exp3_batch.ipynb
| |____WellOpt_trajectory_design.ipynb
|____sample_data
| |____analytic_params.npy
|____pyDGSA_dev
| |____tutorial_detailed.ipynb
| |____plot.py
| |____tutorial_short.ipynb
| |____LICENSE
| |____interact_util.py
| |______init__.py
| |____MANIFEST.in
| |____README.md
| |____dgsa.py
| |____setup.py
| |____cluster.py
|____.gitignore
|____0_demo
| |____CCS_injection_horizon_18n30yrs.ipynb
| |____dev_HM_InSAR_rst_exp2.ipynb
| |____.DS_Store
| |____SC_simfiles.ipynb
| |____Transformer_demo.ipynb
| |____SPR_data_visualization.ipynb
| |____CCS_analysis_singlecase.ipynb
| |______init__.py
| |____demo_H2_gasification_HMcase1.ipynb
| |____CCS_simrst_animation.ipynb
| |____dev_gstats_realization.ipynb
| |____CCS_all_in_npy_orgdata.ipynb
| |____CCS_Petrel_Etchegoin_KHmaps.ipynb
| |____dev_HM_InSAR_rst.ipynb
| |____dev_HM_InSAR_exp2.ipynb
| |____4Elliot_leakage_assessment.ipynb
| |____CCS_CMG_rst2npy.ipynb
| |____demo_H2_MLAI.ipynb
| |____SC_outfiles.ipynb
| |____Petrel_Gslib2npy.ipynb
| |____CCS_injection_purity.ipynb
|____NRAP
| |____fault_leakage_component.py
|____pyCMG_copyright_v1.1.docx
|____AI_utils
| |______init__.py
| |____train.py
|____CMGvsFEM
| |____Case1_simpleFEM
| | |____CMG_output_2D_dip9_shale.xlsx
|____.git
| |____config
| |____objects
| | |____pack
| | | |____pack-923dda2ee4c0f32d80b1a6b9a6bb4e4c4dc50b6e.pack
| | | |____pack-923dda2ee4c0f32d80b1a6b9a6bb4e4c4dc50b6e.idx
| | |____info
| |____HEAD
| |____info
| | |____exclude
| |____logs
| | |____HEAD
| | |____refs
| | | |____heads
| | | | |____main
| | | |____remotes
| | | | |____origin
| | | | | |____HEAD
| |____description
| |____hooks
| | |____commit-msg.sample
| | |____pre-rebase.sample
| | |____pre-commit.sample
| | |____applypatch-msg.sample
| | |____fsmonitor-watchman.sample
| | |____pre-receive.sample
| | |____prepare-commit-msg.sample
| | |____post-update.sample
| | |____pre-merge-commit.sample
| | |____pre-applypatch.sample
| | |____pre-push.sample
| | |____update.sample
| | |____push-to-checkout.sample
| |____refs
| | |____heads
| | | |____main
| | |____tags
| | |____remotes
| | | |____origin
| | | | |____HEAD
| |____index
| |____packed-refs
|____3_RS_monitor_AI
| |____.DS_Store
| |____train_UNet_image2image_parse_file.py
| |____RS_CMG_rst2npy.ipynb
| |____train_UNet_disp2dpres.py
| |____RS_PCA_all_images.ipynb
| |____rst_UNet_PRES_viz_animation.ipynb
| |____RS_InSAR_ML_PCA.ipynb
| |____rst_UNet_viz.ipynb
| |____RS_PCA_corr_maps.ipynb
| |____train_UNet_input.txt
| |____rst_UNet_SG_viz_animation.ipynb
| |____train_UNet_image2image.py
|____HPC_Sherlock
| |____dev_wrt_subm_sherlock.ipynb
| |____dev_run_rwd_sherlock.py
| |____TPL_pycontrol.py
| |____submit.sh
| |____sherlock_submit_jobs.sh
| |____wrt_pyCTRLfiles.ipynb
| |____CMG2npy_on_sherlock.py
|____AI_models
| |______init__.py
| |____UNet_model_v2.py
| |____UNet_model.py
```


## Copyright statement 
Copyright © 2022-2024 SUETRI-A Research Group, The Board of Trustees of the Leland Stanford Junior University.
All rights reserved.
