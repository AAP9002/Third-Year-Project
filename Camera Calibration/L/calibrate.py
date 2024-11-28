import numpy as np
import cv2 as cv
import glob
import os

# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

dataset_path = "/home/aap9002/Stereo-Road-Curvature-Dashcam"
calibration_location = dataset_path+'/camera_calibration/L/calibration.npz'
calibration_images_path = dataset_path+'/camera_calibration/L/*.jpg'

if os.path.exists(calibration_location):
    data = np.load(calibration_location)
    mtx = data['mtx']
    dist = data['dist']

else:
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((9*6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    images = glob.glob(calibration_images_path)

    print(images)
    
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (9,6), None)
    
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
    
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
    
            # Draw and display the corners
            cv.drawChessboardCorners(img, (9,6), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(500)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    np.savez(calibration_location, mtx=mtx, dist=dist)

def get_output_file_name(file):
    """get the output file name

    Args:
        file (string): input file path which is used to determine the output file path

    Returns:
        string: new file path with "L_calibrated.mov"
    """
    return os.path.join(os.path.dirname(file), 'L_calibrated.mov')

def check_if_file_exists(file):
    """check if the file exists

    Args:
        file (string): file path

    Returns:
        bool: True if file exists
    """
    return os.path.exists(get_output_file_name(file))

def get_file_names():
    """gets all the files in the directory that do not have a corresponding calibrated output file

    Returns:
        list: list of file paths to be processed
    """
    day_files = os.listdir(dataset_path+'/day')
    day_files = [os.path.join(dataset_path+'/day', f) for f in day_files if os.path.isdir(os.path.join(dataset_path+'/day', f))]

    source_video_file_name = 'L.MP4'
    day_files = [os.path.join(f, source_video_file_name) for f in day_files]

    night_files = os.listdir(dataset_path+'/night')
    night_files = [os.path.join(dataset_path+'/night', f) for f in night_files if os.path.isdir(os.path.join(dataset_path+'/night', f))]

    source_video_file_name = 'L.MP4'
    night_files = [os.path.join(f, source_video_file_name) for f in night_files]

    files = day_files + night_files
    print(f'checking {len(files)} files')

    # remove if already processed
    files = [f for f in files if not check_if_file_exists(get_output_file_name(f))]
    print(f'processing {len(files)} files')

    files = sorted(files)

    return files

files = get_file_names()

for file in files:
    print(f'Processing {file}')
    output_file = get_output_file_name(file)

    cap = cv.VideoCapture(file)
    ret, first_frame = cap.read()
    h,  w = first_frame.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    rec = cv.VideoWriter(output_file, cv.VideoWriter_fourcc(*'XVID'), 30, (w, h))

    # cv.namedWindow('frame', cv.WINDOW_NORMAL)

    # read video and un-distort frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # undistort
        dst = cv.undistort(frame, mtx, dist, None, newcameramtx)

        # Crop and resize to original dimensions
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        dst = cv.resize(dst, (frame.shape[1], frame.shape[0]))

        rec.write(dst)

        # cv.imshow('frame', dst)
        # if cv.waitKey(1) & 0xFF == ord('q'):
        #     break

    rec.release()

cv.destroyAllWindows()