#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 08:15:45 2021

@author: eilers
"""

import os
import numpy as np
import multiprocessing

os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def execute(process):
    os.system(f'python {process}')
    

def main():
    N = 10
    idx = np.linspace(0, 76, N).astype('int') 
    cmd = 'script_sgplvm.py'
    cmd += ' {} {}'
    cmd = [cmd.format(q0, q1) for q0, q1 in zip(idx[:-1], idx[1:])]
    process_pool = multiprocessing.Pool(processes = N)
    process_pool.map(execute, cmd)
    
    
if __name__ == '__main__':
    main()
    