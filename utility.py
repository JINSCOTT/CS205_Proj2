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


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_accuracy_trace(trace_log, title='Feature Selection Trace', save_path=None, direction='forward'):

 
    accuracies = [acc for _, acc in trace_log]
    # Why doesn't it reverse the x-axis?
    if direction != 'forward':
        x_values = list(reversed(range(len(trace_log), 0, -1)))
    else:
        x_values = list(range(1, len(trace_log)+1))

    # Determine changed feature per step
    feature_changes = []
    for i in range(len(trace_log)):
        current_set = set(trace_log[i][0])
        if i == 0:
            changed = list(current_set)[0]
        else:
            prev_set = set(trace_log[i-1][0])
            diff = current_set.symmetric_difference(prev_set)
            changed = list(diff)[0] if diff else -1
        feature_changes.append(changed)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, accuracies, marker='o')
    for x, y, fidx in zip(x_values, accuracies, feature_changes):
        plt.text(x, y + 0.002, str(fidx), ha='center', fontsize=14, fontweight='bold')

    plt.title(title)
    xlabel = 'Number of Features Selected' if direction == 'forward' else 'Number of Features Removed'
    plt.xlabel(xlabel)

    plt.ylabel('Accuracy')
    plt.grid(True)

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    if save_path:
        plt.savefig(save_path)

    plt.tight_layout()
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


        