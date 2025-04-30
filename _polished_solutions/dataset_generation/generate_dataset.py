import os
import cv2
import numpy as np
import pandas as pd

VIDEO_BANK = '/mnt/Video Bank'
DATASET_PATH = '/home/aap9002/Downloads/RGB_OF_LARGE (with rvps) (Copy)'

Main_Cam_Calibration_path='main_cam_calibration.npz'

MIN_FRAME = 0
SAMPLE_VIDEO_LENGTH = 60

STANDARD_RESOLUTION = (3840, 2160) # 4K resolution

distances = [10,20,30,40,50,75,100]

# Using clusters.csv and average r-vp data we generate our datasest
## WIDE RGB
## NARROW RGB
## WIDE OPTICAL FLOW
## NARROW OPTICAL FLOW

MTX = None
DIST = None
NEW_CAMERA_MTX = None
ROI = None

if os.path.exists(Main_Cam_Calibration_path):
    data = np.load(Main_Cam_Calibration_path)
    MTX = data['mtx']
    DIST = data['dist']

    w = STANDARD_RESOLUTION[0]
    h = STANDARD_RESOLUTION[1]
    NEW_CAMERA_MTX, ROI = cv2.getOptimalNewCameraMatrix(MTX, DIST, (w, h), 1, (w, h))

    print("Camera calibration loaded")

def undistort(img, newcameramtx=NEW_CAMERA_MTX, mtx=MTX, dist=DIST, roi=ROI):
    """
    Undistort the image using the camera calibration parameters.
    """
    if mtx is None or dist is None:
        return img
    else:
        # print("Undistorting image")
        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        dst = cv2.resize(dst, (img.shape[1], img.shape[0]))
        return dst


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

def get_median_vp_if_exists(folder, end_frame):
    # print(f"Checking for median vp in {folder} for {int(end_frame)-20}_{int(end_frame)}_median_vp.txt")
    for item in os.listdir(folder):
        full_path = os.path.join(folder, item)
        suffix = f"{int(end_frame)-20}_{int(end_frame)}_median_vp.txt"
        # print(f"Checking {full_path}")
        if os.path.isdir(full_path):
            rvp = get_median_vp_if_exists(full_path, end_frame)
            if rvp is not None:
                return rvp
            else:
                continue
       
        elif os.path.basename(full_path) == suffix:
            contents = open(full_path, 'r').readlines()
            if len(contents) > 0:
                x = float(contents[0].split(',')[0].strip())
                y = float(contents[0].split(',')[1].strip())
                return (x, y)
    return None

def get_frame_count(video_path):
    """Get the number of frames in a video

    Args:
        video_path (str): The path to the video

    Returns:
        int: The number of frames in the video
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def clip_around_roi(frame, roi):
    x = int(roi[0])
    y = int(roi[1])

    scale = 0.5

    frame_width = int(frame.shape[1])
    frame_height = int(frame.shape[0])

    roi_target_width = int(frame_width * scale)
    roi_target_height = int(frame_height * scale)

    roi_x = int(x - roi_target_width / 2)
    roi_y = int(y - roi_target_height / 2)

    roi_x = max(0, roi_x)
    roi_y = max(0, roi_y)
    roi_x = min(frame_width - roi_target_width, roi_x)
    roi_y = min(frame_height - roi_target_height, roi_y)

    roi_frame = frame[roi_y:roi_y + roi_target_height, roi_x:roi_x + roi_target_width]

    return roi_frame


def resize_frame_to_224(frame):
    """Resize the frame to 224x224

    Args:
        frame (np.ndarray): The frame

    Returns:
        np.ndarray: The resized frame
    """
    # crop 20% off left and right
    frame = frame[:, int(frame.shape[1] * 0.2):int(frame.shape[1] * 0.8)]

    # crop 25% off top and bottom 
    frame = frame[int(frame.shape[0] * 0.25):int(frame.shape[0] * 0.75), :]

    return cv2.resize(frame, (224, 224))

def get_dense_optic_flow_u_v_flow(initial_frame, next_frame): 
    initial_frame_gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
    next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # reduce image size
    initial_frame_gray = cv2.resize(initial_frame_gray, (int(initial_frame_gray.shape[1] * 0.5), int(initial_frame_gray.shape[0] * 0.5)))
    next_frame_gray = cv2.resize(next_frame_gray, (int(next_frame_gray.shape[1] * 0.5), int(next_frame_gray.shape[0] * 0.5)))

    flow = cv2.calcOpticalFlowFarneback(initial_frame_gray, next_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    u = flow[..., 0]
    v = flow[..., 1]

    scale = 0.5

    u = u * scale
    v = v * scale

    mid = 255//2

    u = (u + mid).astype(np.uint8)
    v = (v + mid).astype(np.uint8)

    # hard clipping
    u = np.clip(u, 0, 255)
    v = np.clip(v, 0, 255)

    return u, v

def save_rgb_avi_from_to(INPUT_FILE, OUTPUT_FILE, from_frame, to_frame, rvp_record=None):
    """Save a video from a given frame to another

    Args:
        path (str): The path to the video
        from_frame (int): The starting frame
        to_frame (int): The ending frame
    """

    cap = cv2.VideoCapture(INPUT_FILE, cv2.CAP_FFMPEG)

    ret, sample_frame = cap.read()

    if not ret:
        raise ValueError(f"Could not read frame {from_frame} from {INPUT_FILE}")
    
    sample_frame = undistort(sample_frame)
    
    if rvp_record is not None:
        sample_frame = clip_around_roi(sample_frame, rvp_record)

    sample_frame = resize_frame_to_224(sample_frame)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    print(f"image shape: {sample_frame.shape}")

    out = cv2.VideoWriter(OUTPUT_FILE, fourcc, 15.0, (int(sample_frame.shape[1]), int(sample_frame.shape[0])))

    cap.set(cv2.CAP_PROP_POS_FRAMES, from_frame)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = undistort(frame)
            if rvp_record is not None:
                frame = clip_around_roi(frame, rvp_record)
            frame = resize_frame_to_224(frame)
            out.write(frame)
        else:
            break

        ret, frame = cap.read() # skip a frame

        if cap.get(cv2.CAP_PROP_POS_FRAMES) >= to_frame:
            break

    cap.release()
    out.release()

def save_optic_flow_avi_from_to(INPUT_FILE, OUTPUT_FILE, from_frame, to_frame, max_frame, rvp_record=None):
    """Save a video from a given frame to another

    Args:
        path (str): The path to the video
        from_frame (int): The starting frame
        to_frame (int): The ending frame
    """

    cap = cv2.VideoCapture(INPUT_FILE, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_POS_FRAMES, from_frame)

    # compensate for the skipped frame
    to_frame = min(to_frame+1, max_frame)

    last_frame = cap.read()[1]

    last_frame = undistort(last_frame)

    if rvp_record is not None:
        last_frame = clip_around_roi(last_frame, rvp_record)

    sample_frame = resize_frame_to_224(last_frame)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    print(f"image shape: {sample_frame.shape}")

    out = cv2.VideoWriter(OUTPUT_FILE, fourcc, 15.0, (int(sample_frame.shape[1]), int(sample_frame.shape[0])))

    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = undistort(frame)
            if rvp_record is not None:
                frame = clip_around_roi(frame, rvp_record)

            next_frame = frame          
            u_change, v_change = get_dense_optic_flow_u_v_flow(last_frame, next_frame)

            new_image = np.ones((u_change.shape[0], u_change.shape[1], 3), dtype=np.uint8)
            new_image[..., 0] = u_change
            new_image[..., 1] = v_change

            new_image = resize_frame_to_224(new_image)

            last_frame = next_frame

            # set last channel to grayscale of original frame
            frame = resize_frame_to_224(frame)
            new_image[..., 2] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


            out.write(new_image)
        else:
            break

        ret, frame = cap.read() # skip a frame

        if cap.get(cv2.CAP_PROP_POS_FRAMES) >= to_frame:
            break

    cap.release()
    out.release()



def process_cluster_df(df, related_video_path, output_dir_path):
    TOTAL_FRAME_COUNT = get_frame_count(related_video_path)

    # INIT OUTPUT FOLDERS
    RGB_WIDE_bend_output_folder = os.path.join(output_dir_path, 'RGB_wide')
    RGB_NARROW_bend_output_folder = os.path.join(output_dir_path, 'RGB_narrow')
    OPTIC_FLOW_WIDE_bend_output_folder = os.path.join(output_dir_path, 'Optic_flow_wide')
    OPTIC_FLOW_NARROW_bend_output_folder = os.path.join(output_dir_path, 'Optic_flow_narrow')

    if not os.path.exists(RGB_WIDE_bend_output_folder):
        os.makedirs(RGB_WIDE_bend_output_folder)
    else:
        print(f"Directory {RGB_WIDE_bend_output_folder} already exists")
        RGB_WIDE_bend_output_folder = None
    
    if not os.path.exists(RGB_NARROW_bend_output_folder):
        os.makedirs(RGB_NARROW_bend_output_folder)
    else:
        print(f"Directory {RGB_NARROW_bend_output_folder} already exists")
        RGB_NARROW_bend_output_folder = None

    if not os.path.exists(OPTIC_FLOW_WIDE_bend_output_folder):
        os.makedirs(OPTIC_FLOW_WIDE_bend_output_folder)
    else:
        print(f"Directory {OPTIC_FLOW_WIDE_bend_output_folder} already exists")
        OPTIC_FLOW_WIDE_bend_output_folder = None
    
    if not os.path.exists(OPTIC_FLOW_NARROW_bend_output_folder):
        os.makedirs(OPTIC_FLOW_NARROW_bend_output_folder)
    else:
        print(f"Directory {OPTIC_FLOW_NARROW_bend_output_folder} already exists")
        OPTIC_FLOW_NARROW_bend_output_folder = None

    # process bends of the video
    for i, row in df.iterrows():
        print(f"Processing bend {i}")

        distance_frames = []

        for distance in distances:
            distance_frames.append(row[f'{distance} meters frame'])

        if all([frame == -1 for frame in distance_frames]):
            print(f"Skipping bend {i} as all frames are -1")
            continue

        _direction = "left" if row['avg_angle'] > 0 else "right"
        _speed = row['avg_speed']
        _avg_angle = row['avg_angle']

        # os.path.join(samples_output_folder, f"bend_{i}_{start_frame}_{end_frame}_{direction}_{speed}_{avg_angle}_{distance}_meters_before.avi")
        for s_i in range(len(distance_frames)):
            sample_end_frame = distance_frames[s_i]
            sample_distance = distances[s_i]

            if sample_end_frame == -1:
                print(f"Skipping bend {i} as distance {sample_distance} is -1")
                continue

            sample_start_frame = sample_end_frame - SAMPLE_VIDEO_LENGTH
            
            if sample_end_frame > TOTAL_FRAME_COUNT:
                print(f"Skipping bend {i} as end frame {sample_end_frame} is greater than total frame count {TOTAL_FRAME_COUNT}")
                continue

            # check if rvp exists
            rvp_record = get_median_vp_if_exists(output_dir_path, sample_end_frame)
            print(f"RVP record: {rvp_record}")

            sample_file_name = f"bend_{i}_{sample_start_frame}_{sample_end_frame}_{_direction}_{_speed}_{_avg_angle}_{sample_distance}_meters_before.avi"

            if RGB_WIDE_bend_output_folder is not None:
                sample_file_path_rgb = os.path.join(
                    RGB_WIDE_bend_output_folder,
                    sample_file_name
                )
                print(f"Saving sample Wide RGB video to {sample_file_path_rgb}")
                save_rgb_avi_from_to(related_video_path, sample_file_path_rgb, sample_start_frame, sample_end_frame)


            if OPTIC_FLOW_WIDE_bend_output_folder is not None:
                sample_file_path_optic_flow = os.path.join(
                    OPTIC_FLOW_WIDE_bend_output_folder,
                    sample_file_name
                )
                print(f"Saving sample Wide optic flow video to {sample_file_path_optic_flow}")
                save_optic_flow_avi_from_to(related_video_path, sample_file_path_optic_flow, sample_start_frame, sample_end_frame, TOTAL_FRAME_COUNT)
            
            if rvp_record is None:
                print(f"Skipping narrow video for bend {i} as rvp record is None")
                continue

            if RGB_NARROW_bend_output_folder is not None:
                sample_file_path_rgb = os.path.join(
                    RGB_NARROW_bend_output_folder,
                    sample_file_name
                )
                print(f"Saving sample Narrow RGB video to {sample_file_path_rgb}")
                save_rgb_avi_from_to(related_video_path, sample_file_path_rgb, sample_start_frame, sample_end_frame, rvp_record)

            if OPTIC_FLOW_NARROW_bend_output_folder is not None:
                sample_file_path_optic_flow = os.path.join(
                    OPTIC_FLOW_NARROW_bend_output_folder,
                    sample_file_name
                )
                print(f"Saving sample Narrow optic flow video to {sample_file_path_optic_flow}")
                save_optic_flow_avi_from_to(related_video_path, sample_file_path_optic_flow, sample_start_frame, sample_end_frame, TOTAL_FRAME_COUNT, rvp_record)


def main():
    # Get all video files
    video_file_paths = get_video_files(VIDEO_BANK)
    print(f"Found {len(video_file_paths)} video files")

    # Get all clusters csv file paths
    clusters_csv_file_paths = get_clusters_csv_file_paths(DATASET_PATH)
    print(f"Found {len(clusters_csv_file_paths)} clusters csv files")

    for clusters_csv_file_path in clusters_csv_file_paths:
        print(f"Processing {clusters_csv_file_path}")

        # get direct parent directory of the CSV file
        video_file_name = os.path.dirname(clusters_csv_file_path).split('/')[-1] + '.MP4'
        print(video_file_name)
        related_video_path = None

        # print(video_file_paths)

        # Find the video file path
        for video_file in video_file_paths:
            if video_file_name in video_file:
                related_video_path = video_file
                print(f"found video file: {related_video_path}")
                break
        else:
            print(f"video file not found for {video_file_name}")
            return
    

        if not os.path.exists(related_video_path):
            print(f"Related video {related_video_path} does not exist")
            continue

        # Read the csv file
        df = pd.read_csv(clusters_csv_file_path)

        output_folder_name = os.path.join(os.path.dirname(clusters_csv_file_path))    

        # Process the cluster dataframe
        process_cluster_df(df, related_video_path, output_folder_name)


if __name__ == "__main__":
    main()