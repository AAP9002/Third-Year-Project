import cv2
import os

ROOT_FOLDER = '/home/aap9002/Downloads/RGB_OF_LARGE'
COMBINED_VIDEO_FOLDER = 'COMBINED_samples.MP4'

def resize_frame(frame, width, height):
    # Resize the frame to the specified width and height
    if frame is None:
        raise ValueError("Frame is None, cannot resize.")
    
    # check if required
    if frame.shape[0] != height or frame.shape[1] != width:
        resized_frame = cv2.resize(frame, (width, height))
        return resized_frame
    
    return frame

def get_vid_AVI_file_paths(folder):
    clusters_csv_file_paths = []
    for item in os.listdir(folder):
        full_path = os.path.join(folder, item)
        if os.path.isdir(full_path):
            clusters_csv_file_paths.extend(get_vid_AVI_file_paths(full_path))
        elif full_path.endswith('50_meters_before.avi'):
            clusters_csv_file_paths.append(full_path)
    return clusters_csv_file_paths


get_all_videos = get_vid_AVI_file_paths(ROOT_FOLDER)

video_recorder = cv2.VideoWriter(COMBINED_VIDEO_FOLDER, cv2.VideoWriter_fourcc(*'mp4v'), 30, (448, 448))

narrow_view_rvp = [v for v in get_all_videos if 'RGB_narrow' in v]


for video_path in narrow_view_rvp:
    rgb_narroe_path = video_path
    of_narroe_path = video_path.replace('RGB_narrow', 'Optic_flow_narrow')
    rgb_wide_path = video_path.replace('RGB_narrow', 'RGB_wide')
    of_wide_path = video_path.replace('RGB_narrow', 'Optic_flow_wide')

    # Read the video file
    video1 = cv2.VideoCapture(rgb_narroe_path)
    video2 = cv2.VideoCapture(of_narroe_path)
    video3 = cv2.VideoCapture(rgb_wide_path)
    video4 = cv2.VideoCapture(of_wide_path)

    # Get the number of frames in the video
    num_frames = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))

    # Loop through each frame in the video
    for i in range(num_frames):
        # Read the frame
        success, frame = video1.read()
        if not success:
            break
        success, frame2 = video2.read()
        if not success:
            break
        success, frame3 = video3.read()
        if not success:
            break
        success, frame4 = video4.read()
        if not success:
            break

        wide = cv2.hconcat([frame3, frame4])
        narrow = cv2.hconcat([frame, frame2])
        combined_frame = cv2.vconcat([wide, narrow])

        print(combined_frame.shape)
        # break

        # Resize the frame to 1920x1080
        # resized_frame = resize_frame(frame, 1920, 1080)

        # Write the resized frame to the output video
        video_recorder.write(combined_frame)

    # Release the video capture object
    video1.release()
    video2.release()
    video3.release()
    video4.release()

# Release the video recorder
video_recorder.release()
