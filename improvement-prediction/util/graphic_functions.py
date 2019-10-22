from matplotlib import pyplot as plt

def plot_scatterplot(real_values, predicted_values, image_name, xlabel, ylabel):
    plt.scatter(real_values, predicted_values, alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(image_name, dpi=300)
    plt.close()
