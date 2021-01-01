import matplotlib.pyplot as plt
import numpy as np


x_values = [20, 40, 60, 80, 90, 95]
# ida_taxi_demand_r2_scores = [0.6136339027346756, 0.5993114310515522, 0.41524948366232384, 0.256148936786169, 0.20251711682113716, 0.00010551276604220394]
# containment_taxi_demand_r2_scores = [0.6105482743646506, 0.5727704447529896, 0.3837281676581393, 0.3147933722003907, -0.050717076968390895, -0.06515974990867046]

# ax1 = plt.subplot(211)
# plt.plot(x_values, ida_taxi_demand_r2_scores, marker='o', linestyle='--', color='blue', label='PRIDA')
# plt.plot(x_values, containment_taxi_demand_r2_scores, marker='*', color='red', label='Containment')
# ax1.set_ylabel(r'$R^2$ scores')
# ax1.set_xticks(x_values)
# plt.setp(ax1.get_xticklabels(), visible=False)
# # plt.setp(ax1.get_xticklabels(), fontsize=6)

# # share x only
# ida_taxi_demand_times = np.array([110896.17729187012, 88551.07767105103, 76158.06880950928, 50643.868980407715, 55770.70831298828, 56868.428535461426])
# containment_taxi_demand_times = np.array([47195.41549682617, 60742.29717254639, 33460.76250076294, 22645.75481414795, 15774.297714233398, 11991.324424743652])

# ax2 = plt.subplot(212, sharex=ax1)
# plt.plot(x_values, ida_taxi_demand_times/6000, marker='o', linestyle='--', color='blue', label='PRIDA')
# plt.plot(x_values, containment_taxi_demand_times/6000, marker='*', color='red', label='Containment')
# ax2.set_xticks(x_values)
# ax2.set_ylabel(r'Time ($min$)')
# ax2.set_xlabel(r'Pruning Percentages')
# ax1.legend()
# ax2.legend()
# ax1.grid(True)
# ax2.grid(True)
# ax1.set_title('Efficiency and effectiveness for different pruning percentages\nTaxi Demand use case')
# plt.savefig('taxi_demand_variations_stepwise_linreg.png')

# Taxi -- Linear Regressor + Stepwise
# r2scores = [0.6136339027346756, 0.5993114310515522, 0.41524948366232384, 0.256148936786169, 0.20251711682113716, 0.00010551276604220394]
# candidates = [61, 46, 30, 15, 7, 3]
# times = [110896.17729187012, 88551.07767105103, 76158.06880950928, 50643.868980407715, 55770.70831298828, 56868.428535461426]
# ** containment
# r2scores = [0.6105482743646506, 0.5727704447529896, 0.3837281676581393, 0.3147933722003907, -0.050717076968390895, -0.06515974990867046]
# times = [47195.41549682617, 60742.29717254639, 33460.76250076294, 22645.75481414795, 15774.297714233398,
# 11991.324424743652]
# ** no pruning
# time = 116413.84363174438
# r2score = 0.6683151679569799

ida_taxi_demand_r2_scores = [0.6076477372624748, 0.6470944012755557, 0.3986757862091679, 0.3986757862091679, 0.3986757862091679, 0.3986757862091679]
containment_taxi_demand_r2_scores = [0.6210203341336478, 0.5791372001982557, -0.1786693449103438, -0.1786693449103438, -0.1786693449103438, -0.1786693449103438]
ax1 = plt.subplot(211)
plt.plot(x_values, ida_taxi_demand_r2_scores, marker='o', linestyle='--', color='blue', label='PRIDA')
plt.plot(x_values, containment_taxi_demand_r2_scores, marker='*', color='red', label='Containment')
ax1.set_ylabel(r'$R^2$ scores')
ax1.set_xticks(x_values)
plt.setp(ax1.get_xticklabels(), visible=False)
# plt.setp(ax1.get_xticklabels(), fontsize=6)

# share x only
ida_taxi_demand_times = np.array([3579252.23110199, 2861913.908481598, 1608282.716369629, 1778863.7756156921, 553186.5734672546, 361811.18185043335])
containment_taxi_demand_times = np.array([3678235.704898834, 2578536.9300842285, 2585412.3163223267, 973310.8258247375, 273424.5705604553, 161375.77056884766])

ax2 = plt.subplot(212, sharex=ax1)
plt.plot(x_values, ida_taxi_demand_times/6000, marker='o', linestyle='--', color='blue', label='PRIDA')
plt.plot(x_values, containment_taxi_demand_times/6000, marker='*', color='red', label='Containment')
ax2.set_xticks(x_values)
ax2.set_ylabel(r'Time ($min$)')
ax2.set_xlabel(r'Pruning Percentages')
ax1.legend()
ax2.legend()
ax1.grid(True)
ax2.grid(True)
ax1.set_title('Efficiency and effectiveness for different pruning percentages\nTaxi Demand use case')
plt.savefig('taxi_demand_variations_rifs_linreg.png')

# Taxi -- Linear Regressor + RIFS
# r2scores = [0.6076477372624748, 0.6470944012755557, 0.3986757862091679, 0.3986757862091679, 0.3986757862091679, 0.3986757862091679]
# candidates = [61, 46, 30, 15, 7, 3]
# times = [3579252.23110199, 2861913.908481598, 1608282.716369629, 1778863.7756156921, 553186.5734672546, 361811.18185043335]
# ** containment
# r2scores = [0.6210203341336478, 0.5791372001982557, -0.1786693449103438, -0.1786693449103438, -0.1786693449103438, -0.1786693449103438]
# times = [3678235.704898834, 2578536.9300842285, 2585412.3163223267, 973310.8258247375, 273424.5705604553, 161375.77056884766]
# ** no pruning
# time = 4121130.278110504
# r2score = 0.6895214993628815
