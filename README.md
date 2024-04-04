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


## Copyright statement 
Copyright © 2022-2024 SUETRI-A Research Group, The Board of Trustees of the Leland Stanford Junior University.
All rights reserved.
