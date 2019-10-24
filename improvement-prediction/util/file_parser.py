#!/usr/bin/env python3
from augmentation_instance import *
from constants import *

def parse_augmentation_instances(filename):
    with open(filename, 'r') as f:
        augmentation_instances = []
        for line in f:
            query_filename, target, candidate_filename, r2_score_before, r2_score_after = line.strip().split(SEPARATOR)
            fields = {'query_filename': query_filename,
                      'target_name': target,
                      'candidate_filename': candidate_filename,
                      'r2_score_before': float(r2_score_before),
                      'r2_score_after': float(r2_score_after)}
            augmentation_instances.append(AugmentationInstance(fields))
        return augmentation_instances
