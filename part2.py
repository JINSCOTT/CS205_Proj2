# Note that we expect the spambase dataset to be in the same directory as this script
import numpy as np
import os
from utility import z_score_norm, plot_accuracy_trace
from classifier import forward_selection, backward_elimination

# We create a new function to load spambase dataset
# The delimiter is a comma, not space
def load_and_prepare_spambase(file_path="spambase.data"):
    # Assuming the file is in the current directory or specified path
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Spambase file not found at {file_path}")
    data = np.loadtxt(file_path, delimiter=',')
    labels = data[:, 0].astype(int)
    features = data[:, 1:].astype(float)
    return features, labels

def main():
    print("running part2.py...")
    # Setting
    np.set_printoptions(precision=3, suppress=True)
    # Load and normalize the spambase dataset
    features, labels = load_and_prepare_spambase("spambase.data")
    features = z_score_norm(features)
    
    # Choose algorithm
    print("\nChoose feature selection method:")
    print("1. Forward Selection")
    print("2. Backward Elimination")
    algo_choice = input("Enter option number (1-2): ").strip()

    if algo_choice == '1':
        print("Running Forward Selection...")
        best_subset, trace_log = forward_selection(features, labels)
        title = f'Forward Selection - spambase'
        dir = 'forward'
    elif algo_choice == '2':
        print("Running Backward Elimination...")
        best_subset, trace_log = backward_elimination(features, labels)
        title = f'Backward Elimination - spambase'
        dir = 'backward'
    else: 
        print("Invalid selection method.")
        return
    # Plot the accuracy trace
    print(f"\nFinal best Subset: {best_subset}")
    print("Plotting Accuracy Trace...")
    plot_accuracy_trace(trace_log, title=title,direction=dir)


if __name__ == "__main__":
    main()
    
