import matplotlib.pyplot as plt
import numpy as np

x_values = [20, 40, 60, 80, 90, 95]
ida_college_r2_scores = [0.5322, 0.5322, 0.5322, 0.5172, 0.5041, 0.4054]
containment_college_r2_scores = [0.4732, 0.4732, 0.4632, 0.4344, 0.3213, 0.2818]

ax1 = plt.subplot(211)
plt.plot(x_values, ida_college_r2_scores, marker='o', linestyle='--', color='blue', label='PRIDA')
plt.plot(x_values, containment_college_r2_scores, marker='*', color='red', label='Containment')
ax1.set_ylabel(r'$R^2$ scores')
ax1.set_xticks(x_values)
plt.setp(ax1.get_xticklabels(), visible=False)
# plt.setp(ax1.get_xticklabels(), fontsize=6)

# share x only
ida_college_times = np.log([14292330, 14030510, 14071990, 13367100, 6307350, 4356070])
containment_college_times = np.log([11456160, 11392830, 11337700, 6783070, 6279840, 3173840])

ax2 = plt.subplot(212, sharex=ax1)
plt.plot(x_values, ida_college_times, marker='o', linestyle='--', color='blue', label='PRIDA')
plt.plot(x_values, containment_college_times, marker='*', color='red', label='Containment')
ax2.set_xticks(x_values)
ax2.set_ylabel(r'Time ($\log(ms)$)')
ax2.set_xlabel(r'Pruning Percentages')
ax1.legend()
ax2.legend()
ax1.grid(True)
ax2.grid(True)
ax1.set_title('Efficiency and effectiveness for different pruning percentages\nCollege-debt use case')
plt.savefig('college_variations.png')

