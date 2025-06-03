import numpy as np
from utility import z_score_norm, load_dataset, plot_accuracy_trace


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
        distances = np.linalg.norm(cur_feature - other_features,axis=1)
     
        # We find the index of the closest one, minimum distance
   
        nearest_index = np.argmin(distances)
      
        # Compare and add in result
        predicted_result = other_labels[nearest_index]
        if cur_label == predicted_result:
            correct+=1
    # Calculate accuracy
    accuracy = correct / total
    return accuracy


def forward_selection(features, labels, local_minimum_threshold = 1):
    num_features = len(features[0])
    # Start from empty
    selected_features = [] 
    # best overall
    best_accuracy = 0
    best_feature = None
    # best current round
    current_best_accuracy = 0
    current_best_feature = None
    # logging
    trace_log = []
    # Threshold
    local_minimum = local_minimum_threshold
    print(f'This dataset has {len(features[0])} features(not including the class attribute), with {len(features)} instances')
    print("Begin search:")
    # this controls the number of feature selected, starting from 1 
    for _ in range(num_features):

        current_best_accuracy = 0
        current_best_feature = None

        # Note that it is the "index" of feature not the actual feature
        for feature in range(num_features):
            # Skip out if we already selected this
            if feature in selected_features:
                continue

            # Calculate accuracy
            current_features = selected_features + [feature]
          
            accuracy = nearest_neighbor_classification(features, labels, current_features)
            print(f'Using feature(s) {current_features}, accuracy is {accuracy:.2f}')



            if accuracy > current_best_accuracy:
                current_best_accuracy = accuracy
                current_best_feature = current_features
                # global best must be current best
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_feature = current_features
        
        # Start with the current best next round
        selected_features = current_best_feature
        trace_log.append((selected_features.copy(), current_best_accuracy))

        # End condition
        if current_best_feature!=best_feature and local_minimum == 0:
            # early end
            break
        elif current_best_feature!=best_feature:
            # local minimum
            print(f'The accuracy is decreasing!! Feature set is {current_best_feature}, Accuracy is {current_best_accuracy:.2f}. feature set {best_feature}, was the best. Accuracy is {best_accuracy:.2f}.')
            # Reduce the counter
            local_minimum -= 1
        else:
            # Continue
            print(f'Feature set {best_feature} was the best. Accuracy is {best_accuracy:.2f}')
            # refresh threashoold
            local_minimum = local_minimum_threshold

    print(f'Finished search!! The best feature subset is  {best_feature} which has an accuracy of {best_accuracy:.2f}')
    return best_feature, trace_log


def backward_elimination(features, labels, local_minimum_threshold=1):
    num_features = len(features[0])
    # Start from full
    selected_features = list(range(num_features))
    # Initial values
    best_accuracy = nearest_neighbor_classification(features, labels, selected_features)
    best_feature = selected_features.copy()
    # Threshold
    local_minimum = local_minimum_threshold
    trace_log = [(selected_features.copy(), best_accuracy)]
    # Print begin condition
    print(f'This dataset has {len(features)} records, and {len(features[0])} features')
    print(f'Using feature(s) {best_feature}, accuracy is {best_accuracy:.2f}')
    print("Begin search:")
    while len(selected_features) > 1:

        current_best_accuracy = 0
        feature_to_remove = None
        # Pick out the feature that gives us the highest accuracy
        for feature in selected_features:

            current_features = selected_features.copy()
            current_features.remove(feature)
            accuracy = nearest_neighbor_classification(features, labels, current_features)
            print(f'Using feature(s) {current_features}, accuracy is {accuracy:.2f}')

            if accuracy > current_best_accuracy:
                current_best_accuracy = accuracy
                feature_to_remove = feature
                # global best must be current best
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_feature = current_features
        
        selected_features.remove(feature_to_remove)
        trace_log.append((selected_features.copy(), current_best_accuracy))
        # check
        if current_best_accuracy != best_accuracy and local_minimum==0:
           # early end
           break
        elif current_best_accuracy != best_accuracy:
            print(f'The accuracy is decreasing!! Feature set is {selected_features}, Accuracy is {current_best_accuracy:.2f}. feature set {best_feature}, was the best. Accuracy is {best_accuracy:.2f}.')
            local_minimum -= 1
        else: 
            print(f'Feature set {best_feature} was the best. Accuracy is {best_accuracy:.2f}')
            local_minimum = local_minimum_threshold
    
    print(f'Finished search!! The best feature subset is  {best_feature} which has an accuracy of {best_accuracy:.2f}')
    return best_feature, trace_log
    



# Test functions
if __name__ == "__main__":

    # load dataset
    labels = np.array([1, 1, 2])
    features = np.array([[1,1, 5],[1,2, 5],[10,10,5]])
    selected_features = [0,1, 2]

    acc = nearest_neighbor_classification(features, labels, selected_features)
    print('Accuracy: ', acc)

    # Accuracy is 0.666
    from utility import load_dataset, z_score_norm

    #features, labels = load_dataset('CS205_small_Data__27.txt')
    features, labels = load_dataset('CS205_large_Data__6.txt')
    features = z_score_norm(features)
  
    #forward_selection(features, labels)
    best_subset, trace_log = backward_elimination(features, labels)

    plot_accuracy_trace(trace_log, title='Forward Selection on Small Dataset')