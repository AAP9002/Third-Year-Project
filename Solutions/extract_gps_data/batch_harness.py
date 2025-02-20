from enum import unique
import os

BATCH_FOLDER = '/home/aap9002/Stereo-Road-Curvature-Dashcam/unlabelled/sub'
OUTPUT_FOLDER = '/home/aap9002/Stereo-Road-Curvature-Dashcam/unlabelled/sub'

SCRIPT_PATH = "bend_extract.py"

def get_video_files(folder):
    video_files = []
    for item in os.listdir(folder):
        full_path = os.path.join(folder, item)
        if os.path.isdir(full_path):
            video_files.extend(get_video_files(full_path))
        elif full_path.endswith('.mp4') or full_path.endswith('.MP4'):
            video_files.append(full_path)
    return video_files

def get_file_name_from_path(file_path):
    # get file name without file extention
    return os.path.splitext(os.path.basename(file_path))[0]

def get_unique_output_folder(output_folder, video_file):
    return os.path.join(output_folder, get_file_name_from_path(video_file))

def build_command(video_file, unique_video_output_folder):
    os.makedirs(unique_video_output_folder, exist_ok=True)
    return f"python {SCRIPT_PATH} --input {video_file} --output {unique_video_output_folder}"

def main():
    print("finding video files in folder: ", BATCH_FOLDER)
    video_files = get_video_files(BATCH_FOLDER)

    for video_file in video_files:
        print(f"processing video file: {video_file}")
        unique_video_output_folder = get_unique_output_folder(OUTPUT_FOLDER, video_file)
        if os.path.exists(unique_video_output_folder):
            print(f"skipping video file: {video_file} as output folder already exists")
            continue

        command = build_command(video_file, unique_video_output_folder)
        print(f"running command: {command}")
        os.system(command)

if __name__ == '__main__':
    main()