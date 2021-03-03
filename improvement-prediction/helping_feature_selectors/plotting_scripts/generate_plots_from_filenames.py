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
         if 'Prepruning' in l:
            method = l.strip().split()[-1]  
         if sep_token in l:
             new_section = []
             token = l.strip()
         elif 'MSE' in l:
             sections[token] = new_section
         new_section.append(l)
     return method, sections

def process_section(section, prepruning_method):
     print(prepruning_method)
     entries = {'total_time': 0.0,
                'prepruning_time': 0.0,
                'feature_selection_time': 0.0,
                'user_model_time': 0.0,
                'r2_score': 0.0}
     for line in section:
         if 'time to' in line:
             time_ = float(line.strip().split()[-2])
             entries['total_time'] += time_
             if 'train our model' in line or 'predict what candidates to keep' in line or (prepruning_method != 'ida' and 'perform join' in line):
                  entries['prepruning_time'] += time_
             elif 'run feature selector' in line:
                  entries['feature_selection_time'] += time_
             elif "create user's model with chosen candidates" in line:
                  entries['user_model_time'] += time_
         elif 'R2-' in line:
             entries['r2_score'] = float(line.strip().split()[-1])
     return entries
    
def get_times_and_scores_from_filename(filename, separation_token):
     lines = open(filename).readlines()
     method, sections = create_sections(lines, separation_token)
     keys = {}
     for key in sorted(sections.keys()):
         keys[key] = process_section(sections[key], method)
     return keys

def plot_graphs(prida_dict, containment_dict, use_case, method, image_name, split_times=False):
    labels = list(prida_dict.keys())
    if 'top-' in labels[0]:
        labels = sorted(labels, key=lambda x: int(x.split('-')[1]))
    else:
        labels = sorted(labels)

    prida_r2_scores = []
    containment_r2_scores = []
    prida_times = []
    prepruning_times = []
    feature_selection_times = []
    user_model_times = []
    
    containment_times = []
    cont_prepruning_times = []
    cont_feature_selection_times = []
    cont_user_model_times = []
    for label in labels:
        prida_r2_scores.append(prida_dict[label]['r2_score'])
        containment_r2_scores.append(containment_dict[label]['r2_score'])
        prida_times.append(prida_dict[label]['total_time']/100)
        prepruning_times.append(prida_dict[label]['prepruning_time']/100)
        feature_selection_times.append(prida_dict[label]['feature_selection_time']/100)
        user_model_times.append(prida_dict[label]['user_model_time']/100)
        containment_times.append(containment_dict[label]['total_time']/100)
        cont_prepruning_times.append(containment_dict[label]['prepruning_time']/100)
        cont_feature_selection_times.append(containment_dict[label]['feature_selection_time']/100)
        cont_user_model_times.append(containment_dict[label]['user_model_time']/100)
        
    ax1 = plt.subplot(211)
    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    plt.bar(x - width/2, prida_r2_scores, width, color='red', label='PRIDA')
    plt.bar(x + width/2, containment_r2_scores, width, hatch='\\', edgecolor='blue', fill=False, label='Containment')
    ax1.set_ylabel(r'$R^2$ scores')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    # share x only
    
    ax2 = plt.subplot(212, sharex=ax1)
    if split_times:
         plt.bar(x - width/2, prepruning_times, width, color='green', label='PRIDA')
         plt.bar(x - width/2, feature_selection_times, width, color='brown', bottom=prepruning_times, label='PRIDA')
         plt.bar(x - width/2, user_model_times, width, color='black', bottom=np.array(prepruning_times)+np.array(feature_selection_times), label='PRIDA')

         plt.bar(x + width/2, cont_prepruning_times, width, hatch='\\', edgecolor='green', fill=False, label='Containment')
         plt.bar(x + width/2, cont_feature_selection_times, width, hatch='\\', edgecolor='brown', fill=False,
                 bottom=cont_prepruning_times, label='Containment')
         plt.bar(x + width/2, cont_user_model_times, width, hatch='\\', edgecolor='black', fill=False,
                 bottom=np.array(cont_prepruning_times)+np.array(cont_feature_selection_times), label='Containment')
    else: 
         plt.bar(x - width/2, prida_times, width, color='red', label='PRIDA')
         plt.bar(x + width/2, containment_times, width, hatch='\\', edgecolor='blue', fill=False, label='Containment')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    yticks = ax2.get_yticks()
    
    ax2.set_yticklabels(yticks)
    ax2.set_ylabel('Runtime (s)')
    ax2.set_xlabel(r'Top-N Candidates Kept')
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(color='gray')
    ax2.set_axisbelow(True)
    ax2.yaxis.grid(color='gray')

    ax1.set_title(use_case + '\n' + method)
    plt.tight_layout()
    ax1.legend()
    ax2.legend()
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
    print('**** PRIDA', prida_times_and_scores)
    print('**** CONTAINMENT', containment_times_and_scores)
    plot_graphs(prida_times_and_scores, containment_times_and_scores, use_case, method, image_name, split_times=True)
