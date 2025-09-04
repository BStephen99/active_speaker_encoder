import glob
import csv
import numpy as np
import pandas as pd
from datetime import datetime

def compute_average_precision(precision, recall):
    """Compute Average Precision according to the definition in VOCdevkit."""
    if precision is None:
        if recall is not None:
            raise ValueError("If precision is None, recall must also be None")
        return np.nan

    if not isinstance(precision, np.ndarray) or not isinstance(recall, np.ndarray):
        raise ValueError("precision and recall must be numpy arrays")
    if precision.dtype != float or recall.dtype != float:
        raise ValueError("input must be float numpy array.")
    if len(precision) != len(recall):
        raise ValueError("precision and recall must be of the same size.")
    if not precision.size:
        return 0.0
    if np.amin(precision) < 0 or np.amax(precision) > 1:
        raise ValueError("Precision must be in the range of [0, 1].")
    if np.amin(recall) < 0 or np.amax(recall) > 1:
        raise ValueError("recall must be in the range of [0, 1].")
    if not all(recall[i] <= recall[i + 1] for i in range(len(recall) - 1)):
        raise ValueError("recall must be a non-decreasing array")

    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[0], precision, [0]])

    # Smooth precision to be monotonically decreasing.
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = np.maximum(precision[i], precision[i + 1])

    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    average_precision = np.sum((recall[indices] - recall[indices - 1]) * precision[indices])
    return average_precision

def calculate_precision_recall(df):
    """Calculates precision and recall arrays from the DataFrame."""
    # Ensure the columns are numeric
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df['ground_truth'] = pd.to_numeric(df['ground_truth'], errors='coerce')

    # Drop rows with NaN values
    df = df.dropna(subset=['score', 'ground_truth'])

    if df.empty or df['ground_truth'].sum() == 0:
        # Return arrays of zeros if the dataframe is empty or contains no positive ground truth
        return np.array([0.0]), np.array([0.0]), "null"

    df = df.sort_values(by='score', ascending=False).reset_index(drop=True)

    df['tp'] = (df['ground_truth'] == 1).astype(int).cumsum()
    df['fp'] = (df['ground_truth'] == 0).astype(int).cumsum()

    df['precision'] = df['tp'] / (df.index + 1)
    df['recall'] = df['tp'] / df['ground_truth'].sum()

    precision = np.array(df['precision'])
    recall = np.array(df['recall'])

    # Print precision and recall arrays for debugging
    print("Precision array:", precision)
    print("Recall array before sorting:", recall)

    # Ensure recall is non-decreasing
    recall = np.maximum.accumulate(recall)

    # Print recall array after sorting for debugging
    print("Recall array after sorting:", recall)

    return precision, recall, df['tp'].shape

def calculate_mAP_from_csv(filename):
    """Calculates mAP from the given CSV file."""
    # Load the CSV file
    df = pd.read_csv(filename, header=None)
    df = df[~df.apply(lambda row: row.astype(str).str.contains('pepper', case=False).any(), axis=1)]

    # Ensure the columns of interest are present
    if df.shape[1] < 10:
        raise ValueError("The CSV file must have at least 10 columns.")

    # Extract ground truth and prediction scores
    df = df[[3, 9]]  # Assuming the ground truth is in the 3rd column and the score in the 9th
    df.columns = ['ground_truth', 'score']

    # Calculate precision and recall
    precision, recall, shape = calculate_precision_recall(df)

    # Compute the average precision
    average_precision = compute_average_precision(precision, recall)

    return average_precision, shape

#timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
global_csv_file_path = f"/home2/bstephenson/active-speakers-context/mAP/global_results_{timestamp}.csv"

# Open the global CSV file in write mode
with open(global_csv_file_path, mode='w', newline='') as global_csv_file:
    csv_writer = csv.writer(global_csv_file)
    # Write the header row
    csv_writer.writerow(['CSV File', 'Mean Average Precision (mAP)', "shape"])

    # Loop through all CSV files and calculate mAP
    for g in glob.glob("/home2/bstephenson/active-speakers-context/ours_forward2/*.csv"):
        print(g)
        if "Old" in g or "220927_CLIP_" not in g:
            continue
        mAP, shape = calculate_mAP_from_csv(g)
        print(f"Mean Average Precision (mAP): {mAP}")
        print("")

        # Write the g and mAP to the global CSV file
        csv_writer.writerow([g, mAP, shape[0]])
