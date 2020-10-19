"""
This script gets a file with the number of instances within a range of prediction probabilities in the format 

[probs [0, 0.05), probs [0.95, 1]]

And then generates a bar plot with them
"""

import sys
import matplotlib.pyplot as plt
import numpy as np

probs = eval(open(sys.argv[1]).readline().strip())
ind = np.arange(len(probs))                                                               
#width = 0.35 

fig, ax = plt.subplots()                                                                        
rects = ax.bar(ind, probs, color='#1d3557')                                                 
ax.set_ylabel('Percentages', fontsize=16) #, fontweight='bold')                                                                    
ax.set_xlabel('Intervals of Prediction Probability', fontsize=16) #, fontweight='bold')
ax.set_xticks(ind)
ax.set_xticklabels(('[0.0,0.05)', '[0.05,0.1)', '[0.1,0.15)', '[0.15,0.2)', '[0.2,0.25)', '[0.25,0.3)', 
		    '[0.3,0.35)', '[0.35,0.4)', '[0.4,0.45)', '[0.45,0.5)', '[0.5,0.55)', '[0.55,0.6)', 
		    '[0.6,0.65)', '[0.65,0.7)', '[0.7,0.75)', '[0.75,0.8)', '[0.8,0.85)', '[0.85,0.9)', 
		    '[0.9,0.95)', '[0.95,1.0]'))
for label in ax.get_xticklabels():
        label.set_rotation(90)
        label.set_fontsize(10)
plt.tight_layout()
plt.savefig('pred_probs_plot.png', dpi=600)                                                                                                               

