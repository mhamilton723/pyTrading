from __future__ import print_function
__author__ = 'Mark'
import pandas as pd
import numpy as np
import random
import os

base_dir = '~/machine_learning/stock_sandbox'
script_dir = 'scripts'

base_file = "#!/bin/bash \n\
#PBS -q {cluster} \n\
#PBS -N {name} \n\
#PBS -l nodes=1:ppn=8,mem=5gb \n\
#PBS -l walltime={time} \n\
#PBS -j oe \n\
cd $PBS_O_WORKDIR \n\
python {file} --model_name {model} --dataset {dataset} > {log}"

models = ['shallow_RNN', 'shallow_LSTM', 'shallow_GRU', 'deep_RNN', 'deep_LSTM', 'deep_GRU']
datasets = ['jigsaw', 'synthetic', 'sp500']
file_names = []
batch_files = []
for model in models:
	for dataset in datasets:
		time, cluster = ('71:00:00', 'fas_long') #if model.startswith('deep') else ('23:00:00', 'fas_normal')
		name = model+'_'+dataset
		args = {'cluster': cluster,
				'name': name,
				'time': time,
				'file': base_dir+'/stock_rnn.py',
				'model': "'" + model + '"',
				'dataset': "'" + dataset + '"',
				'log': base_dir+'/'+script_dir+'/' +name+'_log.txt' }

		file_names.append(name+ '_run.pbs')
		batch_files.append(base_file.format(**args))

os.chdir(os.path.expanduser(base_dir))
if not os.path.exists(script_dir):
	os.makedirs(script_dir)

for batch_file, file_name in zip(batch_files, file_names):
    with open(script_dir + '/' + file_name, 'w+') as f: 
        print(batch_file, file=f)

os.chdir(script_dir)
for file_name in file_names:
	command = 'qsub ' + file_name
	os.system(command)
