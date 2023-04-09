#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 16:50:29 2023

@author: anmol
"""

import polars as pl
import numpy as np
from tqdm.notebook import tqdm
import os, sys, pickle, glob, gc
from collections import Counter
import itertools
print("we will use polars", pl.__version__)

def read_file(f):
    df = pl.read_csv(f)
    print(df.columns)
    df['ts'] = (df['ts'] / 1000).astype(int32)
    print(df.columns)
    df['type'] = df['type'].map(type_labels).astype('int8')
    return df

data_cache = {}
type_labels = {'clicks': 0, 'carts': 1, 'orders': 2}
files = glob.glob('../input/otto-chunk-data-inparquet-format/*_parquet/*')
for f in files: data_cache[f] = read_file(f)

READ_CT = 5
CHUNK = int(np.ceil(len(files) / 6))
print(f"We will process {len(files)} files, in groups of {READ_CT} and chunks of {CHUNK}.")

type_weight = {0: 1, 1: 6, 2: 3}

DISK_PIECES = 4
SIZE = 1.86e6/DISK_PIECES

for PART in range(DISK_PIECES):
    print("\n### Disk Part", PART + 1)
    
    for j in range(6):
        a = j * CHUNK
        b = min((j+1) * CHUNK,len(files))
        print(f"Processing files {a} through {b-1} in groups of {READ_CT}")
        
        for k in range(a, b, READ_CT):
            df = [read_file(files[k])]
            for i in range(1, READ_CT):
                if k + i < b: df.append(read_file(files[k+i]))
            df = pl.concat(df, ignore_index=True, axis=0)
            df = df.sort_values(['session', 'ts'], ascending=[True, False])
            
            df = df.reset_index(drop=True)
            df['n'] = df.groupby('session').cumcount()
            df = df.loc[df['n'] < 30].drop('n', axis=1)
            
            df = df.merge(df, on='session')
            df = df.loc[ ((df['ts_x'] - df['ts_y']).abs() < 24 * 60 * 60) * (df['aid_x'] != df['aid_y']) ]
            
            df = df.loc[ (df['aid_x'] >= PART * SIZE) & (df['aid_x'] < (PART + 1) * SIZE) ]
            
            df = df[['session', 'aid_x', 'aid_y', 'type_y']].drop_duplicates(['session', 'aid_x', 'aid_y'])
            df['wgt'] = df['type_y'].map(type_weight)
            
            df = df[['aid_x', 'aid_y', 'wgt']]
            df['wgt'] = df['wgt'].astype('float32')
            df = df.groupby(['aid_x', 'aid_y'])['wgt'].sum()
            
            if k == a:
                tmp2 = df
            else:
                tmp2 = tmp2.add(df, fill_value=0)
            print(k, ', ', end="")
        print()
        
        if a == 0:
            tmp = tmp2
        else:
            tmp = tmp.add(tmp2, fill_value=0)
        del tmp2, df
        gc.collect()
        
    tmp = tmp.reset_index()
    tmp = tmp.sort_values(['aid_x', 'wgt'], ascending=[True, False])
    
    tmp = tmp.reset_index(drop=True)
    tmp['n'] = tmp.groupby('aid_x')['aid_y'].cumcount()
    
    tmp = tmp.loc[tmp['n' < 15]].drop('n', axis=1)
    tmp.to_pandas().to_parquet(f'top_15_carts_orders_{PART}.pqt')
