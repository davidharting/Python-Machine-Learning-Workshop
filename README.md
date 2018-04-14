# Python Machine Learning Workshop
Attending this workshop on 2018 April 16 at Indy.code().

## Environment

### Initial setup
**Do not need to repeat this.**
Recording how I initially setup the environment

First, download Anaconda [here](https://www.anaconda.com/download/#download). Choose the Python 3.6 version.

Step through the installer. After completion, you now have an `anaconda3` directory at the root of your system. 

On OSX, you do not have an "anaconda prompt" like on windows. Instead, you open a regular old terminal and "activate" Anaconda.

```bash
# Initial environment setup
source /anaconda3/bin/activate
conda create -n python-ml-workshop --clone root
```
Note, that to create a working anaconda environment, I had to do it from the command line. Using anaconda-navigator did not create working environments for me for some reason.

### Environment activation
**Do this in every new terminal.**

Activate the correct anaconda environment
```bash
# Activate Anaconda
source /anaconda3/bin/activate
# Switch to workshop-specific environment
conda activate python-ml-workshop
```





