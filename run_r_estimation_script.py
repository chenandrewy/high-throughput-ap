# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 13:35:41 2024

@author: cdim
"""



import subprocess
import os


    
path_root = os.getcwd() # + '/../../'
path_r_script = path_root + '/'
path_input = path_root + '/../../Data/'
path_output = path_root + '/../../Data/'



r_script = path_r_script + 'estimate_EB_distr.r'
# r_script = path_r_script + 'r_trial.r'
rollsignal_name = 'OOS_signal_tstat_OosNyears1.csv.gzip'
out_prefix = 'ChuksDebug_'


#%%

# Command to run R script using Rscript
command = ['Rscript', r_script, 
           '--data_path', path_input, 
           '--rollsignal_name', rollsignal_name, 
           '--out_prefix', out_prefix]
# command = ['Rscript', r_script]



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


