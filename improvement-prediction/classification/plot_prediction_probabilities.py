"""
This script gets a file with the number of instances within a range of prediction probabilities in the format 

[probs [0, 0.1), probs [0.9, 1]]

And then generates a bar plot with them
"""

import sys
import matplotlib.pyplot as plt
import numpy as np

probs = eval(open(sys.argv[1]).readline().strip())
ind = np.arange(len(probs))                                                               
#width = 0.35 

fig, ax = plt.subplots()                                                                        
rects = ax.bar(ind, probs, color='blue')                                                 
ax.set_ylabel('Percentages', fontsize=12, fontweight='bold')                                                                    
ax.set_xlabel('Intervals of Prediction Probability', fontsize=12, fontweight='bold')
ax.set_xticks(ind)
ax.set_xticklabels(('[0.0,0.1)', '[0.1,0.2)', '[0.2,0.3)', '[0.3,0.4)', '[0.4,0.5)', '[0.5,0.6)', '[0.6,0.7)', '[0.7,0.8)', '[0.8,0.9)', '[0.9,1.0]'))
for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_fontsize(8)
plt.tight_layout()
plt.savefig('pred_probs_plot.png', dpi=600)                                                                                                               

