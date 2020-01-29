""" Given (1) two files (one with training and one with validation or test data; both with headers), 
          (2) a threshold alpha above which a gain in R2 squared should correspond to class GOOD GAIN, and 
          (3) a file with the features that should be used for learning, 

          this script explores different ways of combining classification results with other sources of info 
          in order to recommend useful datasets for augmentation. 
"""

#TODO ideas: the evaluation of recommendation should be range-by-range. considering the ordering of results may not be the best idea ever, but may still be relevant
