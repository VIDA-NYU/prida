""" Given (1) a use case dataset with features FEATURES, target TARGET, and column eval indicating whether each instance/row is  a 
          false positive (FP), true positive (TP), false negative (FN), or true negative (TN) according to a certain classifier

          this script generates histograms for every feature and target separated by FP, TP, FN, and TN.
"""

import sys
import pandas as pd

TARGET = 'gain_in_r2_score'
FEATURES = ['query_num_of_columns', 'query_num_of_rows', 'query_row_column_ratio', 'query_max_skewness', 'query_max_kurtosis',
            'query_max_unique', 'candidate_num_of_columns', 'candidate_num_rows', 'candidate_row_column_ratio', 'candidate_max_skewness',
            'candidate_max_kurtosis', 'candidate_max_unique', 'query_target_max_pearson', 'query_target_max_spearman', 'query_target_max_covariance',
            'query_target_max_mutual_info', 'candidate_target_max_pearson', 'candidate_target_max_spearman', 'candidate_target_max_covariance',
            'candidate_target_max_mutual_info', 'max_pearson_difference', 'containment_fraction']

def plot_feature_and_target_histograms(data):
    for feature in FEATURES:
        
    def plot_histogram(hist_filename, feature_name, feature_training, feature_test, remove_outliers_zscores=False, remove_outliers_mad=False):
    if remove_outliers_zscores:
        feature_training = remove_outliers_based_on_zscores(feature_training)
        feature_test = remove_outliers_based_on_zscores(feature_test)
    elif remove_outliers_mad:
        feature_training = remove_outliers_based_on_mad(feature_training)
        feature_test = remove_outliers_based_on_mad(feature_test)

    weights = np.ones_like(feature_training)/float(len(feature_training))
    plt.hist(feature_training, bins=50, alpha=0.5, weights=weights, label='positive-gain')#, normed=True) #density=True, stacked=True)

    weights = np.ones_like(feature_test)/float(len(feature_test))
    plt.hist(feature_test, bins=50, alpha=0.5, weights=weights, label='negative-gain')#, normed=True) #density=True, stacked=True) #, density=True
         
    plt.legend(loc='upper right')
    plt.xlabel('Value Ranges')
    plt.ylabel('Percentages')
    plt.title(feature_name)
    plt.savefig(hist_filename,  dpi=600)
    plt.close()

if __name__=='__main__':
    use_case_dataset = pd.read_csv(sys.argv[1])

    fp = use_case_dataset.loc[use_case_dataset == 'fp']
    plot_feature_and_target_histograms(fp, 'fp')
    tp = use_case_dataset.loc[use_case_dataset == 'tp']
    fn = use_case_dataset.loc[use_case_dataset == 'fn']
    tn = use_case_dataset.loc[use_case_dataset == 'tn']
    
#    num_features = training.shape[1]   
#    for i in range(num_features):
#        plot_histogram(FEATURE_NAMES[i] + '_histograms_training_test.png', FEATURE_NAMES[i], training[:,i], test[:,i], remove_outliers_mad=True) 
#    plot_histogram(FEATURE_NAMES[8] + '_histograms_training_test.png', FEATURE_NAMES[8], training[:,8], test[:,8])
#    plot_histogram(FEATURE_NAMES[12] + '_histograms_training_test.png', FEATURE_NAMES[12], training[:,12], test[:,12])

    # plot_histogram('gains_in_r2_score_histograms_training_test.png', 'gains_in_r2_score', training_gains, test_gains, remove_outliers_mad=True)

    # num_features = training.shape[1]   
    # for i in range(num_features):
    #     feature_for_positive_gain, feature_for_negative_gain = separate_feature_based_on_gain_range(training[:,i], training_gains) 
    #     plot_histogram(FEATURE_NAMES[i] + '_training_corresponding_positive_and_negative_gains.png', FEATURE_NAMES[i], feature_for_positive_gain, feature_for_negative_gain, remove_outliers_mad=True)
    #     feature_for_positive_gain, feature_for_negative_gain = separate_feature_based_on_gain_range(test[:,i], test_gains) 
    #     plot_histogram(FEATURE_NAMES[i] + '_test_corresponding_positive_and_negative_gains.png', FEATURE_NAMES[i], feature_for_positive_gain, feature_for_negative_gain, remove_outliers_mad=True) 
