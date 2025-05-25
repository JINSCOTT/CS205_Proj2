import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Loads the dataset from file path
def load_dataset(file_path):
    data = []
    # Open file
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip(): # make pretty
                values = list(map(float, line.strip().split()))
                data.append(values)
    data = np.array(data)
    # No point to leave label as float, for safety and simplcity
    labels = data[:, 0].astype(int)
    features = data[:, 1:]
    return features, labels

# Feature normalization
# (x - mean)/ std
def z_score_norm(features):
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    result = (features - mean) / std
    return result


def plot_accuracy_trace(trace_log, title='Feature Selection Trace', save_path=None, direction ='forward'):
    steps = list(range(1, len(trace_log)+1))
    accuracies = [acc for _, acc in trace_log]

    plt.figure()
    plt.plot(steps, accuracies, marker='o')
    plt.title(title)
    if direction == 'forward':
        plt.xlabel('Number of Features Selected')
    else:
        plt.xlabel('Number of Features Removed')
    plt.ylabel('Accuracy')
    plt.grid(True)
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    if save_path:
        plt.savefig(save_path)
    plt.show()


# Test functions
if __name__ == "__main__":

    # load dataset
    features, labels = load_dataset("CS205_small_Data__27.txt")

    # do normalization
    features = z_score_norm(features)

    #print

    for i in range(len(labels)):
        print(labels[i], ' ', features[i])


        