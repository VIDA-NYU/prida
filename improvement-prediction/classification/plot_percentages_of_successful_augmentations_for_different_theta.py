'''
This script plots an image showing the number of candidates for each theta, painting the bars to indicate
how the percentage of successful augs increases with theta.
'''

import numpy as np
import matplotlib.pyplot as plt

N = 11

taxi_succ = (447, 66, 22, 18, 18, 18, 18, 18, 18, 18, 6)
taxi_unsucc = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
college_succ = (130, 122, 120, 100, 98, 92, 86, 84, 84, 84, 6)
college_unsucc = (973, 53, 51, 25, 25, 25, 25, 25, 25, 25, 3)
poverty_succ = [np.log(i) for i in (11526, 15, 11, 11, 11, 11, 11, 11, 11, 11, 11)]
poverty_unsucc = [np.log(i) for i in (119402, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0)]
  
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, taxi_succ, width, color='#1d3557')
p2 = plt.bar(ind, taxi_unsucc, width, 
             bottom=taxi_succ, color='#e63946')

plt.ylabel('Candidates')
plt.xlabel(r'Threshold $\theta$')
#plt.title(r'Taxi-Demand -- Candidates for Different $\theta$')
plt.xticks(ind, ('0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'))
#plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0]), ('Successful', 'Unsuccessful'))
plt.tight_layout()
plt.savefig('taxi_demand_varying_theta.png', dpi=600)
plt.close()


p1 = plt.bar(ind, college_succ, width, color='#1d3557')
p2 = plt.bar(ind, college_unsucc, width, 
             bottom=college_succ, color='#e63946')

plt.ylabel('Candidates')
plt.xlabel(r'Threshold $\theta$')
plt.title(r'College-Debt -- Candidates for Different $\theta$')
plt.xticks(ind, ('0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'))
#plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0]), ('Successful', 'Unsuccessful'))
plt.tight_layout()
#plt.savefig('college_debt_varying_theta.png', dpi=600)
plt.close()


p1 = plt.bar(ind, poverty_succ, width, color='#1d3557')
p2 = plt.bar(ind, poverty_unsucc, width, 
             bottom=poverty_succ, color='#e63946')

plt.ylabel(r'Candidates ($\ln$)')
plt.xlabel(r'Threshold $\theta$')
#plt.title(r'Poverty-Estimation -- Candidates for Different $\theta$')
plt.xticks(ind, ('0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'))
#plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0]), ('Successful', 'Unsuccessful'))
plt.tight_layout()
plt.savefig('poverty_estimation_varying_theta.png', dpi=600)
plt.close()
