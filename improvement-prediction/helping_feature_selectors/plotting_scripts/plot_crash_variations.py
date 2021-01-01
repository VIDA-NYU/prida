import matplotlib.pyplot as plt
import numpy as np

x_values = [20, 40, 60, 80, 90, 95]
ida_crash_r2_scores = [0.9691170515958021, 0.9649274101684133, 0.965736237091571, 0.9649797566410734, 0.9726417645516913, 0.9732515795925913]
containment_crash_r2_scores = [0.5969617407747152, 0.5345370430890348, 0.46304474725286715, 0.3183301211414049, 0.35304095587276263, 0.05413232925765188]

ax1 = plt.subplot(211)
plt.plot(x_values, ida_crash_r2_scores, marker='o', linestyle='--', color='blue', label='PRIDA')
plt.plot(x_values, containment_crash_r2_scores, marker='*', color='red', label='Containment')
ax1.set_ylabel(r'$R^2$ scores')
ax1.set_xticks(x_values)
plt.setp(ax1.get_xticklabels(), visible=False)
# plt.setp(ax1.get_xticklabels(), fontsize=6)

# share x only
ida_crash_times = np.array([25990.21053314209, 19730.383281707764, 18832.772006988525, 18392.82850265503, 17987.223072052002, 18055.146045684814])
containment_crash_times = np.array([34347.195625305176, 32804.622650146484, 24829.435348510742, 20469.29121017456, 18388.445377349854, 17733.24966430664])

ax2 = plt.subplot(212, sharex=ax1)
plt.plot(x_values, ida_crash_times/6000, marker='o', linestyle='--', color='blue', label='PRIDA')
plt.plot(x_values, containment_crash_times/6000, marker='*', color='red', label='Containment')
ax2.set_xticks(x_values)
ax2.set_ylabel(r'Time ($min$)')
ax2.set_xlabel(r'Pruning Percentages')
ax1.legend()
ax2.legend()
ax1.grid(True)
ax2.grid(True)
ax1.set_title('Efficiency and effectiveness for different pruning percentages\nVehicle collision use case')
plt.savefig('crash_variations_stepwise_linreg.png')
plt.close()
# Crash -- Linear Regressor + Stepwise
# r2scores = [0.9691170515958021, 0.9649274101684133, 0.965736237091571, 0.9649797566410734, 0.9726417645516913, 0.9732515795925913]
# candidates = [58, 46, 30, 15, 7, 3]
# times = [25990.21053314209, 19730.383281707764, 18832.772006988525, 18392.82850265503, 17987.223072052002, 18055.146045684814]
# ** containment
# r2scores = [0.5969617407747152, 0.5345370430890348, 0.46304474725286715, 0.3183301211414049, 0.35304095587276263, 0.05413232925765188]
# times = [34347.195625305176, 32804.622650146484, 24829.435348510742, 20469.29121017456, 18388.445377349854, 17733.24966430664]
# ** no pruning
# time = 40054.03757095337
# r2score = 0.9617897237026073

ida_crash_r2_scores = [0.9696469610825932, 0.9696469610825932, 0.9730639531052399, 0.9737635879260208, 0.9722640001534714,  0.9662954949436724]
containment_crash_r2_scores = [0.40601316335388216, 0.44593563916595524, 0.48136595592668197, 0.15991760559135482, 0.22773310205157933, -0.33269406071225527]

ax1 = plt.subplot(211)
plt.plot(x_values, ida_crash_r2_scores, marker='o', linestyle='--', color='blue', label='PRIDA')
plt.plot(x_values, containment_crash_r2_scores, marker='*', color='red', label='Containment')
ax1.set_ylabel(r'$R^2$ scores')
ax1.set_xticks(x_values)
plt.setp(ax1.get_xticklabels(), visible=False)
# plt.setp(ax1.get_xticklabels(), fontsize=6)

# share x only
ida_crash_times = np.array([170535.61153411865, 162653.5412979126, 226241.32177352905, 137746.0164451599, 86237.03544616699, 78501.02422714233])
containment_crash_times = np.array([202683.35819244385, 183310.866355896, 141263.65184783936, 145183.8493347168, 113092.65375137329, 64686.195850372314])

ax2 = plt.subplot(212, sharex=ax1)
plt.plot(x_values, ida_crash_times/6000, marker='o', linestyle='--', color='blue', label='PRIDA')
plt.plot(x_values, containment_crash_times/6000, marker='*', color='red', label='Containment')
ax2.set_xticks(x_values)
ax2.set_ylabel(r'Time ($min$)')
ax2.set_xlabel(r'Pruning Percentages')
ax1.legend()
ax2.legend()
ax1.grid(True)
ax2.grid(True)
ax1.set_title('Efficiency and effectiveness for different pruning percentages\nVehicle collision use case')
plt.savefig('crash_variations_rifs_linreg.png')


# Crash -- Linear Regressor + RIFS
# r2scores = [0.9696469610825932, 0.9696469610825932, 0.9730639531052399, 0.9737635879260208, 0.9722640001534714,  0.9662954949436724]
# candidates = [58, 46, 30, 15, 7, 3]
# times = [170535.61153411865, 162653.5412979126, 226241.32177352905, 137746.0164451599, 86237.03544616699, 78501.02422714233]
# ** containment
# r2scores = [0.40601316335388216, 0.44593563916595524, 0.48136595592668197, 0.15991760559135482, 0.22773310205157933, -0.33269406071225527]
# times = [202683.35819244385, 183310.866355896, 141263.65184783936, 145183.8493347168, 113092.65375137329, 64686.195850372314]
# ** no pruning
# time = 314160.1514816284
# r2score = 0.9741459267782678
