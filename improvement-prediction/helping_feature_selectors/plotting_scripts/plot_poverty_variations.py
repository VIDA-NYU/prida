import matplotlib.pyplot as plt
import numpy as np

x_values = [80, 90, 92.5, 95, 97.5, 99]
ida_poverty_r2_scores = [0.3958, 0.3796, 0.3012, 0.3736, 0.3579, 0.2824]
containment_poverty_r2_scores = [0.3740, 0.3790, 0.3838, 0.3011, 0.3956, 0.2812]

ax1 = plt.subplot(211)
plt.plot(x_values, ida_poverty_r2_scores, marker='o', linestyle='--', color='blue', label='PRIDA')
plt.plot(x_values, containment_poverty_r2_scores, marker='*', color='red', label='Containment')
ax1.set_ylabel(r'$R^2$ scores')
ax1.set_xticks(x_values)
plt.setp(ax1.get_xticklabels(), visible=False)
# plt.setp(ax1.get_xticklabels(), fontsize=6)

# share x only
ida_poverty_times = np.log([254300120, 32724310, 7883800, 4612066, 6165760, 8075300])
containment_poverty_times = np.log([75066390, 47430520, 9107420, 7719260, 4688930, 4048020])

ax2 = plt.subplot(212, sharex=ax1)
plt.plot(x_values, ida_poverty_times, marker='o', linestyle='--', color='blue', label='PRIDA')
plt.plot(x_values, containment_poverty_times, marker='*', color='red', label='Containment')
ax2.set_xticks(x_values)
ax2.set_ylabel(r'Time ($\log(ms)$)')
ax2.set_xlabel(r'Pruning Percentages')
ax1.legend()
ax2.legend()
ax1.grid(True)
ax2.grid(True)
ax1.set_title('Efficiency and effectiveness for different pruning percentages\nPoverty estimation use case')
plt.savefig('poverty_variations.png')

