# How to run CMG on Stanford Sherlock?

- Author: Yunan Li
- Date: May 22th, 2024

## Example 1: you need to run 1 CMG model on Sherlock.

**Attention!!!**
- The following example assumes using CMG GEM.
- Need to change the **pwd_CMG** in **pycontrol.py** if you are using STARS, IMEX, or others.

First, we need to prepare a CMG model, that is the dat file. Let’s name it case0.dat. 
- It is better to test from local to make sure it runs with CMG with no errors.
- Sherlock queue takes time, so it is not the optimal solution to debug CMG model errors on Sherlock when you have alternative choices.

Second, let’s get a .sh file, and name it submit.sh for example. 
- The goal is to communicate with Sherlock to request a node.
- An example file could be found in this directory with the name = submit.sh

Third, we need a pycontrol.py file. 
- This is an additional Python file that I personally use to manage what to do on Sherlock.
- The reason is for pyCCUS, users sometimes may want to run only a few components of the pipeline but not the entire, so it is easier to control it in a Python file compared to a Shell script.
- An example file could be found in this directory with the name = pycontrol.py

Fourth, submit the job to Sherlock. 
- Command line: `sbatch -p serc submit.sh`
- Where: it should happen under the directory of your work on Sherlock.
- -p serc means you are using the partition of serc for submission of the job.
- Expected to see the job submitted to the queue or running immediately if we have enough resources. 

Now, you successfully queue for your first CMG job on Stanford Sherlock. Congratulations!

What next?

- When the simulation finishes, it will just stop, and the node is back to Sherlock for use by others.
- You could also set an alarm, that is to receive an email or a notification after it finishes. This is totally up to you!
- I did not include this function in my work, because I may not want to receive 1000+ emails after an experiment. 
