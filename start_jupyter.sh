#!/bin/bash

# ==========================================================
# TO RUN SCRIPT: 
# >> source ./start_jupyter.sh
# ==========================================================

cd ~/dl_tutorial
git pull origin master
wget http://169.254.169.254/latest/meta-data/public-ipv4 -O public-ipv4
source activate dl_aws 
jupyter notebook --no-browser &
sleep 1 
jupyter notebook list > token
python ./print_address.py
