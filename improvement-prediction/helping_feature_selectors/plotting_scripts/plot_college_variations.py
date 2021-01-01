import matplotlib.pyplot as plt
import numpy as np


# College -- Linear Regressor + Stepwise
# r2scores = [0.6938466707786599, 0.697003629725982, 0.6956568530383691, 0.686661795893189, 0.6932757691125525, 0.6967533814238404]
# candidates = [909, 682, 454, 227, 113, 56]
# times = [2617706.1207199097, 2737469.675254822, 2717906.19890213, 2759924.840660095, 2115144.4891929626, 2118149.0974998474]
# ** containment
# r2scores = [0.6583856919458755, 0.6531877224634579, 0.6351967647167255, 0.6402295965457257, 0.6314880762887405, 0.634737096944398]
# times = [1170729.5393943787, 717177.1907806396, 388340.76166152954, 188916.0704612732, 136130.5594444275, 119031.86798095703]
# ** no pruning
# time = 3369320.4760551453
# r2score = 0.6923252013188196

# College -- Linear Regressor + RIFS
# r2scores = [0.6800237903366395, 0.6759662375675671, 0.6763320807995967, 0.6830111441174823, 0.6663529970979796, 0.679728607402474]
# candidates = [909, 682, 454, 227, 113, 56]
# times = [5068912.122573853, 5314268.225898743, 5017701.671524048, 4703492.708740234, 4500988.466262817, 4770702.708816528]
# ** containment
# r2scores = [0.6699465707354776, 0.6513648193049726, 0.6444297337876102, 0.6518258779150951, 0.6389020386410674, 0.6378073042704524]
# times = [4407932.968139648, 3234335.2794647217, 1842508.716583252, 1212005.90133667, 609358.5324287415, 507171.69523239136]
# ** no pruning
# time = 5776609.071750641
# r2score = 0.6977972926718021



labels = ['20%', '40%', '60%', '80%', '90%', '95%']
prida_rifs_college_r2_scores = [0.6800237903366395, 0.6759662375675671, 0.6763320807995967, 0.6830111441174823, 0.6663529970979796, 0.679728607402474]
containment_rifs_college_r2_scores = [0.6699465707354776, 0.6513648193049726, 0.6444297337876102, 0.6518258779150951, 0.6389020386410674, 0.6378073042704524]

ax1 = plt.subplot(211)
x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

plt.bar(x - width/2, prida_rifs_college_r2_scores, width, hatch='\\', edgecolor='red', fill=False, label='PRIDA') #, ida_college_r2_scores, marker='o', linestyle='--', color='blue', label='PRIDA')
plt.bar(x + width/2, containment_rifs_college_r2_scores, width, color='blue', label='Containment')
#plt.plot(x_values, containment_college_r2_scores, marker='*', color='red', label='Containment')
ax1.set_ylabel(r'$R^2$ scores')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
plt.setp(ax1.get_xticklabels(), visible=False)

#rects1 = ax.bar(x - width/2, men_means, width, label='Men')
#rects2 = ax.bar(x + width/2, women_means, width, label='Women')

# Add some text for labels, title and custom x-axis tick labels, etc.
#ax.set_ylabel('Scores')
#ax.set_title('Scores by group and gender')
#ax1.set_xticks(x)
#ax.set_xticklabels(labels)

# def autolabel(rects):
#     """Attach a text label above each bar in *rects*, displaying its height."""
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')


# autolabel(rects1)
# autolabel(rects2)


# ax1 = plt.subplot(211)
# plt.plot(x_values, ida_college_r2_scores, marker='o', linestyle='--', color='blue', label='PRIDA')
# plt.plot(x_values, containment_college_r2_scores, marker='*', color='red', label='Containment')
# ax1.set_ylabel(r'$R^2$ scores')
# ax1.set_xticks(x_values)
# plt.setp(ax1.get_xticklabels(), visible=False)
# plt.setp(ax1.get_xticklabels(), fontsize=6)

# share x only
prida_rifs_college_times = np.log([2617706.1207199097, 2737469.675254822, 2717906.19890213, 2759924.840660095, 2115144.4891929626, 2118149.0974998474])

containment_rifs_college_times = np.log([1170729.5393943787, 717177.1907806396, 388340.76166152954, 188916.0704612732, 136130.5594444275, 119031.86798095703])

ax2 = plt.subplot(212, sharex=ax1)
plt.bar(x - width/2, prida_rifs_college_times, width, hatch='\\', edgecolor='red', fill=False,  label='PRIDA') #, ida_college_r2_scores, marker='o', linestyle='--', color='blue', label='PRIDA')
plt.bar(x + width/2, containment_rifs_college_times, width,  color='blue', label='Containment')

#plt.plot(x_values, prida_rifs_college_times, marker='o', linestyle='--', color='blue', label='PRIDA')
#plt.plot(x_values, containment_college_times, marker='*', color='red', label='Containment')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
yticks = ax2.get_yticks()
print(yticks)
new_yticks = [int(np.exp(y)/1000) for y in yticks]
ax2.set_yticklabels(new_yticks)
ax2.set_ylabel('Runtime (min) \n (in logscale)')
ax2.set_xlabel(r'Pruning Percentages')
#ax1.legend()
#ax2.legend()
ax1.set_axisbelow(True)
ax1.yaxis.grid(color='gray')
ax2.set_axisbelow(True)
ax2.yaxis.grid(color='gray')
#ax1.grid(True)
#ax2.grid(True)
ax1.set_title('College-debt use case -- RIFS + Linear Regression')
plt.tight_layout()
plt.savefig('college_variations_bars.png')

