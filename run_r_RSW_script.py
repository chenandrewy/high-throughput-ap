# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 13:35:41 2024

@author: cdim
"""



import subprocess
import os, re
import pandas as pd
from glob import glob
from tqdm import tqdm
import inspect
from itertools import product

# Get the directory of the current file
current_filename = inspect.getframeinfo(inspect.currentframe()).filename
current_directory = os.path.dirname(current_filename)
path_root = current_directory + "/../../"

# current_directory = os.getcwd() 
path_r_script = current_directory + '/'

path_input = current_directory + '/../../Data/'



r_script = path_r_script + 'estimate_RSW_control.r'



#%% run script with parameters

alpha = 0.10 # use both 0.10, 0.05
subsetmax = 20
nboot = 2000
stock_weight = 'vw'
IS_N_YEARS = 20 # number of years in in-sample period


for tup in product(['ticker', 'acct', 'pastret'], ['ew', 'vw']):
    signal_data, stock_weight = tup
    for yr in range(1983, 2021):
        sampstart = (yr-IS_N_YEARS) * 100 + 1
        sampend = yr * 100 + 12
    
    
        # Command to run R script using Rscript
        command = ['Rscript', r_script,
                   '--data_path', path_input,
                   '--signal_data', signal_data,
                   '--stock_weight', stock_weight,
                   '--sampstart', str(sampstart),
                   '--sampend', str(sampend),
                   '--min_nmonth', '60',
                   '--nboot', str(nboot),
                   '--alpha', str(alpha),
                   '--gamma', '0.05',
                   '--kstepM_itermax', '200',
                   '--subsetmax', str(subsetmax),
                   '--bisect_itermax', '200']
        
        
        # Now run the r code and print the output 
        process = subprocess.Popen(command, stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, text=True)
        
        # Print the output as it is generated
        print("R Script Output:")
        while True:
            output_line = process.stdout.readline()
            if output_line == '' and process.poll() is not None:
                break
            if output_line:
                print(output_line.strip())
        
        # Check if the subprocess was successful
        if process.returncode == 0:
            print("R Script executed successfully.")
        else:
            print("Error executing R script:")
            print(process.stderr.read())


#%% combined results and save

# # read the parameters for RSW Hurdel
path_input_temp = path_input + '/RSW_result/*.csv'
fnames = glob(path_input_temp)
len(fnames)

df_RSW = []
for fname in tqdm(fnames):
    nm = re.findall(r'\\(DebugRSW_.+.csv)', fname)[0]
    nm_ls = nm.split('_')
    signal_fam = '_'.join(nm_ls[1:3])    
    df = pd.read_csv(fname)
    df['signal_family'] = signal_fam
    df_RSW.append(df)
df_RSW = pd.concat(df_RSW)   
df_RSW['oos_begin_year'] = (df_RSW['sampend']//100 + 1).astype(int)
df_RSW = df_RSW.rename(columns={'h': 'tabs_hurdle'})
df_RSW['signal_family'] = df_RSW['signal_family'].replace(
    {'pastret_ew': 'past_ret_ew', 'pastret_vw': 'past_ret_vw'})

# save compiled data
df_RSW.to_csv(path_input + 'RW_HurdleParams.csv', index=False)



