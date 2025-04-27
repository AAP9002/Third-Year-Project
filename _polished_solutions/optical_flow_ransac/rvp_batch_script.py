import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from sympy import comp

SCRIPT_PATH = "rvp_compiled_batch_program.py"

VIDEO_BANK_PATH = '/mnt/Video Bank'
DATASET_FOLDER = '/home/aap9002/Downloads/RGB_OF_LARGE (Copy)'



def get_video_files(folder):
    video_files = []
    for item in os.listdir(folder):
        full_path = os.path.join(folder, item)
        if os.path.isdir(full_path):
            video_files.extend(get_video_files(full_path))
        elif full_path.endswith('FH.mp4') or full_path.endswith('FH.MP4'):
            video_files.append(full_path)
    return video_files

def get_clusters_csv_file_paths(folder):
    clusters_csv_file_paths = []
    for item in os.listdir(folder):
        full_path = os.path.join(folder, item)
        if os.path.isdir(full_path):
            clusters_csv_file_paths.extend(get_clusters_csv_file_paths(full_path))
        elif full_path.endswith('clusters.csv'):
            clusters_csv_file_paths.append(full_path)
    return clusters_csv_file_paths

def build_command(video_file, unique_video_output_folder, start_frame=0, end_frame=0):
    os.makedirs(unique_video_output_folder, exist_ok=True)
    return f"python {SCRIPT_PATH} --i '{video_file}' --o '{unique_video_output_folder}' --s {start_frame} --e {end_frame} -r"

def process_clusters_from_csv(cluster_csv, video_file_paths, MIN_SPEED=15):
    # Read the CSV file
    columns = ['start_idx','end_idx','avg_angle','n_points','avg_speed','start_frame','10 meters frame','20 meters frame','30 meters frame','40 meters frame','50 meters frame','75 meters frame','100 meters frame']
    cluster_csv_df = pd.read_csv(cluster_csv, usecols=columns)

    # get direct parent directory of the CSV file
    video_file_name = os.path.dirname(cluster_csv).split('/')[-1] + '.MP4'
    print(video_file_name)
    video_file_path = None

    # print(video_file_paths)

    # Find the video file path
    for video_file in video_file_paths:
        if video_file_name in video_file:
            video_file_path = video_file
            print(f"found video file: {video_file_path}")
            break
    else:
        print(f"video file not found for {video_file_name}")
        return
    
    
    output_folder_name = os.path.join(os.path.dirname(cluster_csv), f'_rvp')
    if os.path.exists(output_folder_name):
        print(f"skipping {output_folder_name} as it already exists")
        return
    else:
        print(f"creating {output_folder_name} as it does not exist")
        os.makedirs(output_folder_name, exist_ok=True)

    # Iterate through each row in the DataFrame
    for index, row in cluster_csv_df.iterrows():
        # Get the video file name  is parent directory of the CSV file
        if video_file_path is None:
            print(f"skipping {video_file_name} as video file not found")
            continue

        # check min speed
        if row['avg_speed'] < MIN_SPEED:
            print(f"skipping {video_file_name} as avg speed is less than 10 mph")
            continue

        for distance in [10, 20, 30, 40, 50, 75, 100]:
            # Get the frame number for the current distance

            end_frame = int(row[f'{distance} meters frame'])
            start_frame = end_frame - 20

            if end_frame == -1:
                print(f"skipping distance {distance} for start frame {start_frame} as end frame is -1")
                continue

            if start_frame < 0:
                print(f"skipping distance {distance} for start frame {start_frame} as end frame is negative")
                continue

            # Build and run the command
            command = build_command(video_file_path, output_folder_name, start_frame, end_frame)
            print(f"running command: {command}")
            os.system(command)

def main():
    print("finding video files in folder: ", DATASET_FOLDER)
    video_file_paths = get_video_files(VIDEO_BANK_PATH)
    cluster_csvs = get_clusters_csv_file_paths(DATASET_FOLDER)

    # for cluster_csv in cluster_csvs:
    #     print(f"processing cluster csv file: {cluster_csv}")
    #     process_clusters_from_csv(cluster_csv, video_file_paths)
    #     break

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = { executor.submit(process_clusters_from_csv, csv, video_file_paths): csv
                    for csv in cluster_csvs }
        
        total_jobs = len(futures)
        print(f"total jobs: {total_jobs}")

        completed_jobs = 0

        for future in as_completed(futures):
            csv = futures[future]
            try:
                print(f"processing cluster csv file: {csv}")
                completed_jobs += 1
                print(f"completed jobs: {completed_jobs} of {total_jobs}")            
                future.result()
            except Exception as exc:
                print(f"{csv} failed: {exc}")

    # print(video_file_paths)

if __name__ == '__main__':
    main()