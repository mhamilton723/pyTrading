#!/bin/bash
#PBS -q gputest
#PBS -N stock_rnn
#PBS -l nodes=1:ppn=16,mem=10gb
#PBS -l walltime=55:00:00

cd $PBS_O_WORKDIR
module load GPU/Cuda/6.0

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32  python ~/machine_learning/stock_sandbox/stock_rnn.py > ~/machine_learning/stock_sandbox/scripts/logfile.txt