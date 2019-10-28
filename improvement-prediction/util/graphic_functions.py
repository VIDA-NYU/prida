from matplotlib import pyplot as plt

def plot_scatterplot(real_values, predicted_values, image_name, xlabel, ylabel):
    """Given aligned real_values and predicted_values, this method generates a scatterplot 
    where real values are in the x axis and predicted values are in the y axis
    """
    plt.scatter(real_values, predicted_values, alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(image_name, dpi=300)
    plt.close()
