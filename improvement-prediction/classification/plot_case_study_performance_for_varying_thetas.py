'''
This script plots f-measure values for varying thetas for each case study.
'''

import numpy as np
import matplotlib.pyplot as plt

taxi_fmeasures = [0.69, 0.99, 0.74, 0.67, 0.50, 0.97, 0.88, 0.84, 0.97, 0.94, 0.91]
poverty_fmeasures = [0.18, 0.80, 1.00, 0.95, 1.00, 1.00, 0.90, 1.00, 0.95, 1.00, 1.00]
college_fmeasures = [0.39, 0.83, 0.84, 0.89, 0.91, 0.89, 0.81, 0.88, 0.88, 0.86, 0.80]
xlabels = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(18, 8))
axes.plot(xlabels, taxi_fmeasures, 'o-', linewidth=3, color='#809a41', label='taxi-demand')
axes.plot(xlabels, poverty_fmeasures, 'o--', linewidth=3, color='#1d3557', label='poverty-estimation')
axes.plot(xlabels, college_fmeasures, 'o:', linewidth=3, color='#e63946', label='college-debt')
axes.yaxis.grid(True, alpha=0.3)
axes.yaxis.set_tick_params(labelsize=24)
axes.xaxis.set_tick_params(labelsize=24)
axes.set_xlabel(r'Threshold $\theta$', fontsize=28)
axes.set_ylabel('F-measure', fontsize=28)
axes.legend(loc='lower right', prop={'size':22})
plt.tight_layout()
plt.savefig('case_studies_varying_theta.png')
