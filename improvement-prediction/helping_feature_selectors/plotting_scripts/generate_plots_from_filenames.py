'''
This script gets as input:
  argv[1] => filename1
  argv[2] => filename2
  argv[3] => separating token (can be 'top-', can be '%')
  argv[4] => the use case name (e.g., 'Vehicle_Collision')
  argv[5] => method (e.g., 'RIFS_and_RF')
  argv[6] => the name of the png image

and generates an image comparing the times (in minutes) and r2_scores for different pruning levels (top-N or N%) 
'''

import matplotlib.pyplot as plt
import numpy as np
import sys

def create_sections(lines, sep_token):
     sections = {}
     new_section = []
     for l in lines:
         if sep_token in l:
             new_section = []
             token = l.strip()
         elif 'MSE' in l:
             sections[token] = new_section
         new_section.append(l)
     return sections

def process_section(section):
     total_time = 0.0
     for line in section:
         if 'time to' in line:
             total_time += float(line.strip().split()[-2])
         elif 'R2-' in line:
             r2_score = float(line.strip().split()[-1])
     return total_time, r2_score
    
def get_times_and_scores_from_filename(filename, separation_token):
     lines = open(filename).readlines()
     sections = create_sections(lines, separation_token)
     keys = {}
     for key in sorted(sections.keys()):
         keys[key] = process_section(sections[key])
     return keys

def plot_graphs(prida_dict, containment_dict, use_case, method, image_name):
    labels = list(prida_dict.keys())
    if 'top-' in labels[0]:
        labels = sorted(labels, key=lambda x: int(x.split('-')[1]))
    else:
        labels = sorted(labels)

    prida_r2_scores = []
    containment_r2_scores = []
    prida_times = []
    containment_times = []
    for label in labels:
        prida_r2_scores.append(prida_dict[label][1])
        containment_r2_scores.append(containment_dict[label][1])
        prida_times.append(prida_dict[label][0]/600)
        containment_times.append(containment_dict[label][0]/600)
    
    ax1 = plt.subplot(211)
    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    plt.bar(x - width/2, prida_r2_scores, width, color='red', label='PRIDA')
    #, ida_college_r2_scores, marker='o', linestyle='--', color='blue', label='PRIDA')
    plt.bar(x + width/2, containment_r2_scores, width, hatch='\\', edgecolor='blue', fill=False, label='Containment')
    #plt.plot(x_values, containment_college_r2_scores, marker='*', color='red', label='Containment')
    #plt.legend(loc='upper left')
    ax1.set_ylabel(r'$R^2$ scores')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    # share x only
    ax2 = plt.subplot(212, sharex=ax1)
    plt.bar(x - width/2, prida_times, width, color='red', label='PRIDA')
    #, ida_college_r2_scores, marker='o', linestyle='--', color='blue', label='PRIDA')
    plt.bar(x + width/2, containment_times, width, hatch='\\', edgecolor='blue', fill=False, label='Containment')
    #plt.plot(x_values, prida_rifs_college_times, marker='o', linestyle='--', color='blue', label='PRIDA')
    #plt.plot(x_values, containment_college_times, marker='*', color='red', label='Containment')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    yticks = ax2.get_yticks()
    
    ax2.set_yticklabels(yticks)
    ax2.set_ylabel('Runtime (min)')
    ax2.set_xlabel(r'Top-N Candidates Kept')
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(color='gray')
    ax2.set_axisbelow(True)
    ax2.yaxis.grid(color='gray')

    ax1.set_title(use_case + '\n' + method)
    plt.tight_layout()
    plt.savefig(image_name, dpi=600)
 
if __name__ == '__main__':
    
    filename_prida = sys.argv[1]
    filename_containment = sys.argv[2]
    sep_token = sys.argv[3]
    use_case = sys.argv[4]
    method = sys.argv[5]
    image_name = sys.argv[6]

    prida_times_and_scores =  get_times_and_scores_from_filename(filename_prida, sep_token)
    containment_times_and_scores = get_times_and_scores_from_filename(filename_containment, sep_token)
    plot_graphs(prida_times_and_scores, containment_times_and_scores, use_case, method, image_name)
