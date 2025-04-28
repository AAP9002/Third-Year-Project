import cv2
import os

ROOT_FOLDER = '/home/aap9002/Downloads/RGB_OF_LARGE (with rvps) (Copy)'
COMBINED_VIDEO_FOLDER = 'COMBINED_rvp.MP4'

def resize_frame(frame, width, height):
    # Resize the frame to the specified width and height
    if frame is None:
        raise ValueError("Frame is None, cannot resize.")
    
    # check if required
    if frame.shape[0] != height or frame.shape[1] != width:
        resized_frame = cv2.resize(frame, (width, height))
        return resized_frame
    
    return frame

def get_VP_AVI_file_paths(folder):
    clusters_csv_file_paths = []
    for item in os.listdir(folder):
        full_path = os.path.join(folder, item)
        if os.path.isdir(full_path):
            clusters_csv_file_paths.extend(get_VP_AVI_file_paths(full_path))
        elif full_path.endswith('_vp.avi'):
            clusters_csv_file_paths.append(full_path)
    return clusters_csv_file_paths


get_all_videos = get_VP_AVI_file_paths(ROOT_FOLDER)

video_recorder = cv2.VideoWriter(COMBINED_VIDEO_FOLDER, cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1080))

for video_path in get_all_videos:
    # Read the video file
    video = cv2.VideoCapture(video_path)

    # Get the number of frames in the video
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print (f"Processing video: {video_path} with {num_frames} frames.")

    # Loop through each frame in the video
    for i in range(num_frames):
        # Read the frame
        success, frame = video.read()

        if not success:
            break

        # Resize the frame to 1920x1080
        resized_frame = resize_frame(frame, 1920, 1080)

        # Write the resized frame to the output video
        video_recorder.write(resized_frame)

    # Release the video capture object
    video.release()
