import os
import pandas as pd

ROOT = '/home/aap9002/Downloads/RGB_OF_LARGE (patrial)'

def get_nmea_csv_file_paths(folder):
    nmea_csv_file_paths = []
    for item in os.listdir(folder):
        full_path = os.path.join(folder, item)
        if os.path.isdir(full_path):
            nmea_csv_file_paths.extend(get_nmea_csv_file_paths(full_path))
        elif full_path.endswith('nmea.csv'):
            nmea_csv_file_paths.append(full_path)
    return nmea_csv_file_paths


def merge_nmea_csv_files(folder):
    nmea_csv_file_paths = get_nmea_csv_file_paths(folder)
    merged_df = pd.DataFrame()

    for csv_file in nmea_csv_file_paths:
        df = pd.read_csv(csv_file)
        merged_df = pd.concat([merged_df, df], ignore_index=True)

    merged_df.to_csv('merged_nmea.csv', index=False)

if __name__ == "__main__":
    # Set the ROOT variable to the path where the nmea.csv files are located
    merge_nmea_csv_files(ROOT)