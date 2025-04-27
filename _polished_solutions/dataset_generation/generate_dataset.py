import os
import pandas as pd

VIDEO_BANK = '/home/aap9002/Downloads/dashcam/VIDEO_BANK'
DATASET_PATH = '/home/aap9002/Downloads/dashcam/VIDEO'

# Using clusters.csv and average r-vp data we generate our datasest
## WIDE RGB
## NARROW RGB
## WIDE OPTICAL FLOW
## NARROW OPTICAL FLOW

### Get video files
def get_video_files(folder):
    video_files = []
    for item in os.listdir(folder):
        full_path = os.path.join(folder, item)
        if os.path.isdir(full_path):
            video_files.extend(get_video_files(full_path))
        elif full_path.endswith('FH.mp4') or full_path.endswith('FH.MP4'):
            video_files.append(full_path)
    return video_files

### Get clusters csv file paths
def get_clusters_csv_file_paths(folder):
    clusters_csv_file_paths = []
    for item in os.listdir(folder):
        full_path = os.path.join(folder, item)
        if os.path.isdir(full_path):
            clusters_csv_file_paths.extend(get_clusters_csv_file_paths(full_path))
        elif full_path.endswith('clusters.csv'):
            clusters_csv_file_paths.append(full_path)
    return clusters_csv_file_paths

