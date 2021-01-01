import matplotlib.pyplot as plt
import numpy as np

x_values = [20, 40, 60, 80, 90, 95]
ida_pickup_r2_scores = [0.0078, 0.0078, 0.0078, 0.0081, 0.0078, 0.0078]
containment_pickup_r2_scores = [0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078]

ax1 = plt.subplot(211)
plt.plot(x_values, ida_pickup_r2_scores, marker='o', linestyle='--', color='blue', label='PRIDA')
plt.plot(x_values, containment_pickup_r2_scores, marker='*', color='red', label='Containment')
ax1.set_xticks(x_values)
ax1.set_ylabel(r'$R^2$ scores')
plt.setp(ax1.get_xticklabels(), visible=False)
# plt.setp(ax1.get_xticklabels(), fontsize=6)

# share x only
ida_pickup_times = np.log([2207660, 2481440, 2580030, 1005240, 2155410, 1792350])
containment_pickup_times = np.log([2162540, 1839370, 1804610, 1310590, 1399320, 1427140])

ax2 = plt.subplot(212, sharex=ax1)
plt.plot(x_values, ida_pickup_times, marker='o', linestyle='--', color='blue', label='PRIDA')
plt.plot(x_values, containment_pickup_times, marker='*', color='red', label='Containment')
ax2.set_xticks(x_values)
ax2.set_ylabel(r'Time ($\log(ms)$)')
ax2.set_xlabel(r'Pruning Percentages')
ax1.legend()
ax2.legend()
ax1.grid(True)
ax2.grid(True)
ax1.set_title('Efficiency and effectiveness for different pruning percentages\nPickup-prediction use case')
plt.savefig('pickup_variations.png')

