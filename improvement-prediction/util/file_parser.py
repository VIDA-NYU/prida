#!/usr/bin/env python3
from augmentation_instance import *
from constants import *


def parse_augmentation_instances(filename):
    with open(filename, 'r') as f:
        augmentation_instances = []
        for line in f:
            augmentation_instances.append(parse_augmentation_instance(line))
        return augmentation_instances


def parse_augmentation_instance(file_record, use_hdfs=False, hdfs_address=None, hdfs_user=None):
    query_filename, target, candidate_filename, r2_score_before, r2_score_after = file_record.strip().split(SEPARATOR)
    fields = {'query_filename': query_filename,
              'target_name': target,
              'candidate_filename': candidate_filename,
              'r2_score_before': float(r2_score_before),
              'r2_score_after': float(r2_score_after)}
    return AugmentationInstance(fields, use_hdfs, hdfs_address, hdfs_user)
