import numpy as np


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

# Test functions
if __name__ == "__main__":

    # load dataset
    features, labels = load_dataset("CS205_small_Data__27.txt")

    # do normalization
    features = z_score_norm(features)

    #print

    for i in range(len(labels)):
        print(labels[i], ' ', features[i])