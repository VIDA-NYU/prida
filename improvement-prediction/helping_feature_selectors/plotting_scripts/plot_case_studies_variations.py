import matplotlib.pyplot as plt
import numpy as np

# x_values = [20, 40, 60, 80, 90, 95]
# ida_case_study_r2_scores = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# containment_case_study_r2_scores = [0.0, 0.0, -0.09712619827690805, -0.6013764106297781, -0.029659869820702678, -0.029659869820702678]

# ax1 = plt.subplot(211)
# plt.plot(x_values, ida_case_study_r2_scores, marker='o', linestyle='--', color='blue', label='PRIDA')
# plt.plot(x_values, containment_case_study_r2_scores, marker='*', color='red', label='Containment')
# ax1.set_ylabel(r'$R^2$ scores')
# ax1.set_xticks(x_values)
# plt.setp(ax1.get_xticklabels(), visible=False)
# # plt.setp(ax1.get_xticklabels(), fontsize=6)

# # share x only
# ida_case_study_times = np.log([180800, 161700, 135040, 132160, 130010, 136300])
# containment_case_study_times = np.log([148360, 160210, 1539580, 900820, 409060, 243150])

# ax2 = plt.subplot(212, sharex=ax1)
# plt.plot(x_values, ida_case_study_times, marker='o', linestyle='--', color='blue', label='PRIDA')
# plt.plot(x_values, containment_case_study_times, marker='*', color='red', label='Containment')
# ax2.set_xticks(x_values)
# ax2.set_ylabel(r'Time ($\log(ms)$)')
# ax2.set_xlabel(r'Pruning Percentages')
# ax1.legend()
# ax2.legend()
# ax1.grid(True)
# ax2.grid(True)
# ax1.set_title('Efficiency and effectiveness for different pruning percentages\nopenml-test (synthetic use case 0)')
# plt.savefig('case_study_0_variations.png')
# plt.close()

# x_values = [20, 40, 60, 80, 90, 95]
# ida_case_study_r2_scores = [0.10121324146796096, 0.15028014200367046, 0.14991963113808726, 0.20652802262948045, 0.1536497652235278, 0.22503950879075474]
# containment_case_study_r2_scores = [0.12852425815072832, 0.07225147875766513, 0.13408694642133712, 0.06755526301461146, 0.05865411458308856, 0.13066612581531334]

# ax1 = plt.subplot(211)
# plt.plot(x_values, ida_case_study_r2_scores, marker='o', linestyle='--', color='blue', label='PRIDA')
# plt.plot(x_values, containment_case_study_r2_scores, marker='*', color='red', label='Containment')
# ax1.set_ylabel(r'$R^2$ scores')
# ax1.set_xticks(x_values)
# plt.setp(ax1.get_xticklabels(), visible=False)
# # plt.setp(ax1.get_xticklabels(), fontsize=6)

# # share x only
# ida_case_study_times = np.log([200630, 283790, 213310, 144140, 131750, 126460])
# containment_case_study_times = np.log([230720, 193220, 147960, 101240, 87740, 81440])

# ax2 = plt.subplot(212, sharex=ax1)
# plt.plot(x_values, ida_case_study_times, marker='o', linestyle='--', color='blue', label='PRIDA')
# plt.plot(x_values, containment_case_study_times, marker='*', color='red', label='Containment')
# ax2.set_xticks(x_values)
# ax2.set_ylabel(r'Time ($\log(ms)$)')
# ax2.set_xlabel(r'Pruning Percentages')
# ax1.legend()
# ax2.legend()
# ax1.grid(True)
# ax2.grid(True)
# ax1.set_title('Efficiency and effectiveness for different pruning percentages\nopenml-test (synthetic use case 1)')
# plt.savefig('case_study_1_variations.png')
# plt.close()


x_values = [20, 40, 60, 80, 90, 95]
ida_case_study_r2_scores = [0.5322443553570289, 0.145011821968413, 0.48043629583885605, 0.4750556151002331, 0.4448471284035439, 0.4576264350711099]
containment_case_study_r2_scores = [0.5446093266822262, 0.4564356626008761, 0.41523408354404834, 0.3797149807140391, 0.3338877076972785, 0.3515626191609006]

ax1 = plt.subplot(211)
plt.plot(x_values, ida_case_study_r2_scores, marker='o', linestyle='--', color='blue', label='PRIDA')
plt.plot(x_values, containment_case_study_r2_scores, marker='*', color='red', label='Containment')
ax1.set_ylabel(r'$R^2$ scores')
ax1.set_xticks(x_values)
plt.setp(ax1.get_xticklabels(), visible=False)
# plt.setp(ax1.get_xticklabels(), fontsize=6)

# share x only
ida_case_study_times = np.log([244120, 156110, 212390, 210430, 189040, 174650])
containment_case_study_times = np.log([176300, 226820, 139470, 109530, 89290, 85530])

ax2 = plt.subplot(212, sharex=ax1)
plt.plot(x_values, ida_case_study_times, marker='o', linestyle='--', color='blue', label='PRIDA')
plt.plot(x_values, containment_case_study_times, marker='*', color='red', label='Containment')
ax2.set_xticks(x_values)
ax2.set_ylabel(r'Time ($\log(ms)$)')
ax2.set_xlabel(r'Pruning Percentages')
ax1.legend()
ax2.legend()
ax1.grid(True)
ax2.grid(True)
ax1.set_title('Efficiency and effectiveness for different pruning percentages\nopenml-test (synthetic use case 1)')
plt.savefig('case_study_1_variations.png')
plt.close()

x_values = [20, 40, 60, 80, 90, 95]
ida_case_study_r2_scores = [0.3438274660495858, 0.3061905112903637, 0.65030774242836826, 0.76096682231927635, 0.86118634828159315,  0.88892599385762736]
containment_case_study_r2_scores = [0.26363585824319813, 0.3496724004892309, 0.681414326411651, 0.7178727496513149, 0.8198583107633829, 0.8397628823683835]

ax1 = plt.subplot(211)
plt.plot(x_values, ida_case_study_r2_scores, marker='o', linestyle='--', color='blue', label='PRIDA')
plt.plot(x_values, containment_case_study_r2_scores, marker='*', color='red', label='Containment')
ax1.set_ylabel(r'$R^2$ scores')
ax1.set_xticks(x_values)
plt.setp(ax1.get_xticklabels(), visible=False)
# plt.setp(ax1.get_xticklabels(), fontsize=6)

# share x only
ida_case_study_times = np.log([194410, 129200, 148450, 178710, 132210, 140250])
containment_case_study_times = np.log([229760, 318000, 212030, 140430, 134770, 84940])

ax2 = plt.subplot(212, sharex=ax1)
plt.plot(x_values, ida_case_study_times, marker='o', linestyle='--', color='blue', label='PRIDA')
plt.plot(x_values, containment_case_study_times, marker='*', color='red', label='Containment')
ax2.set_xticks(x_values)
ax2.set_ylabel(r'Time ($\log(ms)$)')
ax2.set_xlabel(r'Pruning Percentages')
ax1.legend()
ax2.legend()
ax1.grid(True)
ax2.grid(True)
ax1.set_title('Efficiency and effectiveness for different pruning percentages\nopenml-test (synthetic use case 2)')
plt.savefig('case_study_2_variations.png')
plt.close()


x_values = [20, 40, 60, 80, 90, 95]
ida_case_study_r2_scores = [0.4880404230535722, 0.527954684124285, 0.4342375508768159, 0.48829527299285735, 0.4240553511878915, 0.47235424203030596]
containment_case_study_r2_scores = [0.4880404230535722, 0.4065938754696581, 0.46002069725093053, 0.3855480407247891, 0.33508036102007877, 0.47200430246918135]

ax1 = plt.subplot(211)
plt.plot(x_values, ida_case_study_r2_scores, marker='o', linestyle='--', color='blue', label='PRIDA')
plt.plot(x_values, containment_case_study_r2_scores, marker='*', color='red', label='Containment')
ax1.set_ylabel(r'$R^2$ scores')
ax1.set_xticks(x_values)
plt.setp(ax1.get_xticklabels(), visible=False)
# plt.setp(ax1.get_xticklabels(), fontsize=6)

# share x only
ida_case_study_times = np.log([211740, 542920, 397780, 150180, 111130, 85460])
containment_case_study_times = np.log([201220, 184820, 122330, 136770, 163200, 122250])
ax2 = plt.subplot(212, sharex=ax1)
plt.plot(x_values, ida_case_study_times, marker='o', linestyle='--', color='blue', label='PRIDA')
plt.plot(x_values, containment_case_study_times, marker='*', color='red', label='Containment')
ax2.set_xticks(x_values)
ax2.set_ylabel(r'Time ($\log(ms)$)')
ax2.set_xlabel(r'Pruning Percentages')
ax1.legend()
ax2.legend()
ax1.grid(True)
ax2.grid(True)
ax1.set_title('Efficiency and effectiveness for different pruning percentages\nopenml-test (synthetic use case 3)')
plt.savefig('case_study_3_variations.png')
plt.close()

# x_values = [20, 40, 60, 80, 90, 95]
# ida_case_study_r2_scores = [0.4213753784056511, 0.4770221997981837, 0.46600302724520715, 0.09628879919273481, -0.01319192734611474, 0.15651120080726577]
# containment_case_study_r2_scores = [0.3419035317860749, 0.4116286579212918, 0.4197697275479315, 0.3881951564076691, 0.28686508236727304, 0.138666112630859]

# ax1 = plt.subplot(211)
# plt.plot(x_values, ida_case_study_r2_scores, marker='o', linestyle='--', color='blue', label='PRIDA')
# plt.plot(x_values, containment_case_study_r2_scores, marker='*', color='red', label='Containment')
# ax1.set_ylabel(r'$R^2$ scores')
# ax1.set_xticks(x_values)
# plt.setp(ax1.get_xticklabels(), visible=False)
# # plt.setp(ax1.get_xticklabels(), fontsize=6)

# # share x only
# ida_case_study_times = np.log([483970, 418920, 486640, 336010, 265750, 239340])
# containment_case_study_times = np.log([443260, 361310, 452970, 207150, 108170, 89320])
# ax2 = plt.subplot(212, sharex=ax1)
# plt.plot(x_values, ida_case_study_times, marker='o', linestyle='--', color='blue', label='PRIDA')
# plt.plot(x_values, containment_case_study_times, marker='*', color='red', label='Containment')
# ax2.set_xticks(x_values)
# ax2.set_ylabel(r'Time ($\log(ms)$)')
# ax2.set_xlabel(r'Pruning Percentages')
# ax1.legend()
# ax2.legend()
# ax1.grid(True)
# ax2.grid(True)
# ax1.set_title('Efficiency and effectiveness for different pruning percentages\nopenml-test (synthetic use case 5)')
# plt.savefig('case_study_5_variations.png')
# plt.close()

x_values = [20, 40, 60, 80, 90, 95]
ida_case_study_r2_scores = [0.33403460944582986, 0.3203208485553567, 0.2705308424319667, 0.10591500721480152, 0.24271543134140605, 0.13351972130214584]
containment_case_study_r2_scores = [0.29297540292641544, 0.19312812855113104, 0.27586128256392595, 0.14837834816613849, 0.19409319107046108, 0.12739460088417967]

ax1 = plt.subplot(211)
plt.plot(x_values, ida_case_study_r2_scores, marker='o', linestyle='--', color='blue', label='PRIDA')
plt.plot(x_values, containment_case_study_r2_scores, marker='*', color='red', label='Containment')
ax1.set_ylabel(r'$R^2$ scores')
ax1.set_xticks(x_values)
plt.setp(ax1.get_xticklabels(), visible=False)
# plt.setp(ax1.get_xticklabels(), fontsize=6)

# share x only
ida_case_study_times = np.log([362230, 305550, 203530, 165260, 101430, 84460])
containment_case_study_times = np.log([152670, 156080, 156720, 146800, 133480, 121960])
ax2 = plt.subplot(212, sharex=ax1)
plt.plot(x_values, ida_case_study_times, marker='o', linestyle='--', color='blue', label='PRIDA')
plt.plot(x_values, containment_case_study_times, marker='*', color='red', label='Containment')
ax2.set_xticks(x_values)
ax2.set_ylabel(r'Time ($\log(ms)$)')
ax2.set_xlabel(r'Pruning Percentages')
ax1.legend()
ax2.legend()
ax1.grid(True)
ax2.grid(True)
ax1.set_title('Efficiency and effectiveness for different pruning percentages\nopenml-test (synthetic use case 4)')
plt.savefig('case_study_4_variations.png')
plt.close()

x_values = [20, 40, 60, 80, 90, 95]
ida_case_study_r2_scores = [0.35442316764625814, 0.3807117143534766, 0.6255332699117506, 0.5793029112550416, 0.5806723395816511, 0.5902536109171548]
containment_case_study_r2_scores = [0.5041761661621285, 0.36703981763677274, 0.4600290040885612, 0.3700716163359845, 0.2951247204201124, 0.24269026869077825]
ax1 = plt.subplot(211)
plt.plot(x_values, ida_case_study_r2_scores, marker='o', linestyle='--', color='blue', label='PRIDA')
plt.plot(x_values, containment_case_study_r2_scores, marker='*', color='red', label='Containment')
ax1.set_ylabel(r'$R^2$ scores')
ax1.set_xticks(x_values)
plt.setp(ax1.get_xticklabels(), visible=False)
# plt.setp(ax1.get_xticklabels(), fontsize=6)

# share x only
ida_case_study_times = np.log([278610, 310490, 474460, 832570, 366610, 281340])
containment_case_study_times = np.log([212790, 147290, 183150, 132980, 168700, 131010])
ax2 = plt.subplot(212, sharex=ax1)
plt.plot(x_values, ida_case_study_times, marker='o', linestyle='--', color='blue', label='PRIDA')
plt.plot(x_values, containment_case_study_times, marker='*', color='red', label='Containment')
ax2.set_xticks(x_values)
ax2.set_ylabel(r'Time ($\log(ms)$)')
ax2.set_xlabel(r'Pruning Percentages')
ax1.legend()
ax2.legend()
ax1.grid(True)
ax2.grid(True)
ax1.set_title('Efficiency and effectiveness for different pruning percentages\nopenml-test (synthetic use case 5)')
plt.savefig('case_study_5_variations.png')
plt.close()
