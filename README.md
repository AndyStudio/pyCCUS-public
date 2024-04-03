# pyCCUS public
This repository is designed by Yunan Li from SUETRI-A research group at Stanford University. 


### Note to users
- The public version of pyCCUS reserves functionalities due to the requirements of collaborators for multiple projects as our research moves forward.
- Please feel free to reach out if you need additional support, and we are happy to help :)
- Contact: yunanli@stanford.edu / ylistanford@gmail.com

### pyCCUS overview


pyCCUS aims to automate the large number of simulations needed to analyze Geological Carbon Storage (GCS) outcomes. The goal is to support the CUSP project for CCUS (Carbon Capture, Utilization, and Storage). This toolbox interacts with the commercial software CMG so that the numerical model is parsed by CMG for computations. Our toolbox extracts the results from CMG for post-processing and analysis. 


![Fig4](https://github.com/AndyStudio/pyCCUS-public/assets/39730681/3cb6de68-d3f6-47e4-aaf4-238e030d4ad9)

The overview of this workflow with essential components is noted. At the beginning, we need to build the base simulation model. We may also need the inputs from users as an optional choice if the goal is to drive pyCCUS to create many reservoir simulation models. For example, users could provide a list of parameters with ranges or distributions, and the design of experiment (DoE) algorithms in pyCCUS take over the task to generate the DoE experiment set-up. Therefore, our toolbox writes a number of simulation model input files accordingly. This function is applicable for uncertainty quantification or global sensitivity analysis. We also implemented a novel module to generate deviated injector trajectories for optimization. When the numerical simulation models are created according to user needs, the next step is to run all simulations. The toolbox is capable to deploy the work in different environments, including the Windows operating system (e.g.  your local machine or workstation), Linux operating system, multiple HPC resources (e.g. Stanford Sherlock). The toolbox submits all jobs prioritizing parallel over sequential computations depending on the computational resources allocated.

The numerical simulation results are extracted from CMG to Python for post-processing and analysis. The “sanity check” component ensures the success of the simulation and filters out cases that do not satisfy our settings. For example, the actual injection profile may not be the same as the pre-defined injection scheme due to the well bottom hole pressure (BHP) constraint. For GCS specific tasks, we highlighted the 2D/3D CO2 delineation, CO2 migration distances with uncertainties, case design optimization, and potential AI applications. The CO2 migration distance is essential information for induced seismicity assessments (Kohli et al., 2023). In terms of optimization, pyCCUS is capable for both forward and looped optimization. The looped optimization is favored when doing a history match using a relatively simple model, because the optimizer needs to learn from previous iterations to propose a better set of values. Example cases include chemical kinetics history matching with experimental measurements (Li et al., 2023). The forward optimization reduces the computational cost when we are capable to request enough computational resources to simulate all cases in parallel so that the amount of time needed for all cases is the same as the time needed for one simulation. If the looped optimization is applied, however, there are at least a few iterations required for optimization and the total computational cost is expected to be many times larger than the forward optimization.

### Test to update a gif or mov


https://github.com/AndyStudio/pyCCUS-public/assets/39730681/ea6f7410-09db-4457-8dc2-a00951170be5

