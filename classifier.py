import numpy as np


# Main algorith
# features shoould be numpy arr of features
# labels: numpy array
# feature_selection: selected array index. Starting from 0
def nearest_neighbor_classification(features, labels, feature_selection):
    total = len(features)
    correct = 0

    # Gate features to only selected
    selected_features = features[:, feature_selection]

    # For each object
    for i in range(total):
        # Current obj
        cur_label = labels[i]
        cur_feature = selected_features[i]

        # Others, remove current object form array
        other_labels = np.delete(labels, i, axis=0)
        other_features = np.delete(selected_features, i, axis=0)
        # Nearest neigbor
        # This calculates the distance 
        distances = np.linalg.norm(cur_feature - other_features, axis=1)
        # We find the index of the closest one, minimum distance
        nearest_index = np.argmin(distances)

        # Compare and add in result
        predicted_result = other_labels[nearest_index]
        if cur_label == predicted_result:
            correct+=1
    # Calculate accuracy
    accuracy = correct / total
    return accuracy


# Test functions
if __name__ == "__main__":

    # load dataset
    labels = np.array([1, 1, 2])
    features = np.array([[1,1],[1,2],[10,10]])
    selected_features = [0,1]

    acc = nearest_neighbor_classification(features, labels, selected_features)
    print('Accuracy: ', acc)

    # Accuracy is 0.666
    # This is due 