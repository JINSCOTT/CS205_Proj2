# Small 27, Large 6

import os
import argparse
import numpy as np
from utility import load_dataset, z_score_norm, plot_accuracy_trace
from classifier import forward_selection, backward_elimination


# Choose dataset based on user input or predefined options
def choose_dataset(dataset_select):
    default_files = {
        "small": "CS205_small_Data__27.txt",
        "large": "CS205_large_Data__6.txt"
    }

    if dataset_select in default_files:
        file_path = default_files[dataset_select]
    else:
        # File not found
        if not os.path.isfile(dataset_select):
            print(f"File not found: {dataset_select}")
            exit(1)
        # Should not happen though
        file_path = dataset_select

    print(f"Using dataset: {file_path}")
    return file_path

def main():
    # Setting
    np.set_printoptions(precision=3, suppress=True)

    # Argument parser for command line options
    parser = argparse.ArgumentParser(description="Run Feature Selection with GUI Options")
    parser.add_argument('--dataset', type=str, choices=['small', 'large'], help="Choose predefined dataset")
    parser.add_argument('--file', type=str, help="Path to custom dataset file")
    args = parser.parse_args()
    dir = 'forward'

    # Choose dataset
    if args.file:
        file_path = choose_dataset(args.file)
    elif args.dataset:
        file_path = choose_dataset(args.dataset)
    else:
        print("Choose dataset:")
        print("1. CS205_small_Data__27.txt")
        print("2. CS205_large_Data__6.txt")
        print("3. Custom file path")
        choice = input("Enter option number (1-3): ").strip()
        if choice == '1':
            file_path = choose_dataset("small")
        elif choice == '2':
            file_path = choose_dataset("large")
        elif choice == '3':
            file_path = input("Enter full path to .txt dataset: ").strip()
            file_path = choose_dataset(file_path)
        else:
            # Error selection
            print("Invalid choice.")
            return

    # Load and normalize dataset
    features, labels = load_dataset(file_path)
    features = z_score_norm(features)

    # Choose algorithm
    print("\nChoose feature selection method:")
    print("1. Forward Selection")
    print("2. Backward Elimination")
    algo_choice = input("Enter option number (1-2): ").strip()

    if algo_choice == '1':
        print("Running Forward Selection...")
        best_subset, trace_log = forward_selection(features, labels)
        title = f'Forward Selection - {os.path.basename(file_path)}'
    elif algo_choice == '2':
        print("Running Backward Elimination...")
        best_subset, trace_log = backward_elimination(features, labels)
        title = f'Backward Elimination - {os.path.basename(file_path)}'
        dir = 'backward'
    else:
        print("Invalid selection method.")
        return

    # Final result
    print(f"\nFinal Best Subset: {best_subset}")
    print(f"Plotting Accuracy Trace...")
    plot_accuracy_trace(trace_log, title=title,direction=dir)

if __name__ == "__main__":
    main()