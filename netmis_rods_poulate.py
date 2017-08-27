# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 11:48:35 2016

@author: HarryMunro

This script pulls in raw NETMIS data and RODS data. 
It then does the following:
- Filters out unnecessary stations from a station exclusions list.
- Calculates dwell times.
- Optionally filters dwell times.
- Excludes unwanted lines.
- Matches passenger numbers with stations and time of day.
- Adds SAF rate for each line.
- Adds train density for each line.
- Adds standing capacity for each line.
- Adds seating capacity for each line.
- Calculates availability of each line.
- Saves the new dataset.

Also populates line names.
"""

import pandas as pd
import numpy as np
import time
start_time = time.time()

netmis_file = 'November 2015 NETMIS Data Dates Version.csv'

#df = pd.read_csv(netmis_file, usecols = ['TIMESTAMP', 
#'ACTUAL DEPARTURE TIME', 'SUTOR CODE', 'LINE ID', 'DIRECTION CODE'], 
#nrows = 100000) # with nrows filter

df = pd.read_csv(netmis_file, usecols = ['TIMESTAMP', 
'ACTUAL DEPARTURE TIME', 'SUTOR CODE', 'LINE ID', 'DIRECTION CODE']) # with nrows filter

original = len(df)

# read in the station exclusions list
exclusions_file = 'station_exclusions.txt'
exclusions_list = pd.read_table(exclusions_file, index_col = 'Sutor')
# convert to dictionary
exclusions_list = exclusions_list.to_dict()['Include']
# map the exclusions list to sutor codes
df['INCLUDE'] = df['SUTOR CODE'].map(exclusions_list)
# delete rows containing "n"
df = df[df['INCLUDE'].str.contains('n') == False]

# first need to convert to datetimes
# and drop rows where there is no dwell time data
df = df.dropna()
#df = df.dropna(subset = ['TIMESTAMP'])
#df = df.dropna(subset = ['ACTUAL DEPARTURE TIME'])
#df = df.dropna(subset = ['SUTOR CODE'])
#df = df.dropna(subset = ['LINE ID'])
#df = df.dropna(subset = ['DIRECTION CODE'])
#del df['DWELL TIME']

# convert timestamp data
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format = '%d/%m/%Y %H:%M:%S')
df['ACTUAL DEPARTURE TIME'] = pd.to_datetime(df['ACTUAL DEPARTURE TIME'], format = '%d/%m/%Y %H:%M:%S')

# calculating dwell times - need to convert timestamp data first
df['DWELL TIME'] = df['ACTUAL DEPARTURE TIME'] - df['TIMESTAMP']
df['SUTOR DIRECTION LINE'] = df['SUTOR CODE'] + df['DIRECTION CODE'].map(str) + df['LINE ID'].map(str)

# Cleaning up again
df = df[df['DWELL TIME'] < pd.Timedelta('00:02:00')] # less than
df = df[df['DWELL TIME'] > pd.Timedelta('00:00:05')] # greater than

# deleting lines where we do not have RODs data
df = df[df['LINE ID'] != 11]
df = df[df['LINE ID'] != 14]

#names = list(df.columns.values)
col_list = ['TIMESTAMP', 'DWELL TIME', 'SUTOR CODE', 'LINE ID', 'DIRECTION CODE', 'SUTOR DIRECTION LINE']
df = df[col_list]

# convert dwell times to integers
df['DWELL TIME'] = df['DWELL TIME'].dt.seconds

df = df.reset_index() # resets the index from 

# now import the Alighters data
alighters_filename = 'Alighters.csv'
alighters_df = pd.read_csv(alighters_filename, skiprows = 2)
column_headers = list(alighters_df.columns.values)

# field which column timestamp falls into
alighters = [] # empty list to subsequently merge with dataframe
errors = []
match_errors = []
resampled_column_headers = column_headers[16:]
for row in range(len(df)):
    sample_time = df.TIMESTAMP[row]
    h = sample_time.hour
    m = sample_time.minute

    # convert to strings
    if h < 10:
        h = '0' + str(h)
    else:
        h = str(h)
    
    if m < 10:
        m = '0' + str(m)
    else:
        m = str(m)
   
    match = []
    for n in resampled_column_headers:
        x = int(n[7:9]) # this is to remove the error of int(m[7:9]) equalling 0
        if x == 0:
            x = 59
        if n[0:2] == h and int(m) > int(n[2:4]) and int(m) <= x:
            match.append(n) # match contains the column heading for the match
        
    if len(match) > 1:
        print('WARNING, MULTIPLE MATCHES FOUND')
        
    if len(match) == 0:
        match_errors.append([sample_time, row])

    sample_s_d_l = df['SUTOR DIRECTION LINE'][row]
    # test if sample_s_d_l exits in the alighters database
    #x = alighters_df['Sutor Direction Line'] == sample_s_d_l
    try:
        location = alighters_df[alighters_df['Sutor Direction Line'] == sample_s_d_l].index[0]
        alighters.append(alighters_df[match[0]][location])
    except IndexError:
        errors.append([sample_s_d_l])
        alighters.append(np.nan)
        
df['ALIGHTERS'] = alighters   

# Now join the boarders data to dataset 
boarders_filename = 'Boarders.csv'
boarders_df = pd.read_csv(boarders_filename, skiprows = 2)
column_headers_boarders = list(boarders_df.columns.values)

# field which column timestamp falls into
boarders = [] # empty list to subsequently merge with dataframe
resampled_column_headers = column_headers_boarders[17:]
for row in range(len(df)):
    sample_time = df.TIMESTAMP[row]
    h = sample_time.hour
    m = sample_time.minute

    # convert to strings
    if h < 10:
        h = '0' + str(h)
    else:
        h = str(h)
    
    if m < 10:
        m = '0' + str(m)
    else:
        m = str(m)
   
    match = []
    for n in resampled_column_headers:
        x = int(n[7:9]) # this is to remove the error of int(m[7:9]) equalling 0
        if x == 0:
            x = 59
        if n[0:2] == h and int(m) > int(n[2:4]) and int(m) <= x:
            match.append(n) # match contains the column heading for the match
        
    if len(match) > 1:
        print('WARNING, MULTIPLE MATCHES FOUND')
        
    if len(match) == 0:
        match_errors.append([sample_time, row])

    sample_s_d_l = df['SUTOR DIRECTION LINE'][row]
    # test if sample_s_d_l exits in the alighters database
    #x = alighters_df['Sutor Direction Line'] == sample_s_d_l
    try:
        location = boarders_df[boarders_df['Sutor Direction Line'] == sample_s_d_l].index[0]
        boarders.append(boarders_df[match[0]][location])
    except IndexError:
        errors.append([sample_s_d_l])
        boarders.append(np.nan)
        
df['BOARDERS'] = boarders   

df = df.dropna()


# calculate boarders and alighters
df['BOARDERS AND ALIGHTERS'] = df['BOARDERS'] + df['ALIGHTERS']

# populate with line names
line_names = {0:'Bakerloo', 2:'Central', 3:'Victoria', 4:'Metropolitan', 5:'Northern', 6:'Jubilee', 7:'Piccadilly', 8:'District', 13:'Circle Hammersmith & City'}
df['LINE NAME'] = df['LINE ID'].map(line_names)

# populate with failure data - safs per day per line
saf_rate = {'Bakerloo':6.3886121, 'Central': 14.29395018, 'Victoria': 6.844128113879, 'Metropolitan': 7.92918149466192, 'Northern': 13.0053380782918, 'Jubilee': 12.6241992882562, 'Piccadilly': 9.60747330960854, 'District': 9.30747330960854, 'Circle Hammersmith & City': 10.5996441281139}
df['SAF RATE'] = df['LINE NAME'].map(saf_rate)

# populate with train density (how many trains we have per station/km)
train_density = {'Bakerloo': 1.077586207, 'Central': 0.662162162, 'Victoria': 0.761904762, 'Metropolitan': 114.2941176, 'Northern': 122.96, 'Jubilee': 84.46666667, 'Piccadilly': 115.2075472}
df['TRAIN DENSITY'] = df['LINE NAME'].map(train_density)

# populate with standing capacity per train
standing_capacity = {'Bakerloo': 116.6, 'Central': 155.02, 'Victoria': 153.2, 'Metropolitan': 174, 'Northern': 110.36, 'Jubilee': 145.92, 'Piccadilly': 114}
df['STANDING CAPACITY'] = df['LINE NAME'].map(standing_capacity)

# populate with seating capacity per train
seating_capacity = {'Bakerloo': 268, 'Central': 272, 'Victoria': 252, 'Metropolitan': 306, 'Northern': 200, 'Jubilee': 234, 'Piccadilly': 228}
df['SEATING CAPACITY'] = df['LINE NAME'].map(seating_capacity)

# populate with line-direction availability
df['LINE DIRECTION'] = df['LINE NAME'] + df['DIRECTION CODE'].map(str)
availability = {'Bakerloo0': 0.992752, 'Bakerloo1': 0.9941, 'Central0': 0.9842, 'Central1':0.98, 'Victoria0': 0.9916, 'Victoria1': 0.9927, 'Metropolitan0': 0.9881, 'Metropolitan1': 0.9849, 'Northern0': 0.988, 'Northern1': 0.9879, 'Jubilee0': 0.9863, 'Jubilee1': 0.9884, 'Piccadilly0': 0.9885, 'Piccadilly1': 0.987}
df['LINE DIRECTION AVAILABILITY'] = df['LINE DIRECTION'].map(availability)

# Drop the district, circle and H&S
df = df.dropna()

# Data statistics
print(original - len(df))
print(len(df)/original)
print(time.time() - start_time)


df.to_csv('Populated NETMIS (including days) with RODS.csv')