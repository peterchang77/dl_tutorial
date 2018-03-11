#!/bin/bash

# ==========================================================
# TO RUN SCRIPT: 
# >> source ./start_jupyter.sh
# ==========================================================

cd ~/dl_tutorial
git pull origin master
source activate dl_aws 
jupyter notebook --no-browser
