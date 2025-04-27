import os
import pandas as pd

ROOT = '/home/aap9002/Downloads/RGB_OF_LARGE (patrial)'

def get_clusters_csv_file_paths(folder):
    clusters_csv_file_paths = []
    for item in os.listdir(folder):
        full_path = os.path.join(folder, item)
        if os.path.isdir(full_path):
            clusters_csv_file_paths.extend(get_clusters_csv_file_paths(full_path))
        elif full_path.endswith('clusters.csv'):
            clusters_csv_file_paths.append(full_path)
    return clusters_csv_file_paths


def merge_clusters_csv_files(folder):
    clusters_csv_file_paths = get_clusters_csv_file_paths(folder)
    merged_df = pd.DataFrame()

    for csv_file in clusters_csv_file_paths:
        df = pd.read_csv(csv_file)
        merged_df = pd.concat([merged_df, df], ignore_index=True)

    merged_df.to_csv(os.path.join(folder, 'merged_clusters.csv'), index=False)

if __name__ == "__main__":
    # Set the ROOT variable to the path where the clusters.csv files are located
    merge_clusters_csv_files(ROOT)