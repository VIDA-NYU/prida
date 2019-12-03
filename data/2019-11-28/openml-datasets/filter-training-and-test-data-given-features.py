"""
Given two files designating which data should be used as training and test, 
with each line in the format

<target>,<query>,<candidates>

and a file with computed features,  where each line is in the format

<query>,<target>,<candidate>,<features>,<improvement values>

this script separates the data into training and test sets, writing the records into two different files 
"""

import sys

training_lines = open(sys.argv[1]).readlines()
test_lines = open(sys.argv[2]).readlines()
data_lines = open(sys.argv[3]).readlines()

training_keys = set()
for l in training_lines:
    fields = l.strip().split(',')
    target = fields[0]
    query = fields[1]
    candidates = fields[2:]
    for c in candidates:
        key = target + '-' + query + '-' + c
        training_keys.add(key)
        
test_keys = set()
for l in test_lines:
    fields = l.strip().split(',')
    target = fields[0]
    query = fields[1]
    candidates = fields[2:]
    for c in candidates:
        key = target + '-' + query + '-' + c
        test_keys.add(key)

f_training = open('training-' + sys.argv[3], 'w')
f_test = open('test-' + sys.argv[3], 'w')
for l in data_lines:
    fields = l.strip().split(',')
    key = fields[1] + '-' + fields[0] + '-' + fields[2]
    if key in training_keys:
        f_training.write(l)
    elif key in test_keys:
        f_test.write(l)

f_training.close()
f_test.close()
