# %% [markdown]
# # Extract GPS data from dashcam footage
# gps_bend_finding_with_gaussian_smoothing_and_DBSCAN_clustering_and_circle_fitting

# %%
ids = [43]
min_string_length = 9
DATASET_PATH = "/home/aap9002/Stereo-Road-Curvature-Dashcam"
min_speed_filter = 5

# %% [markdown]
# ### Pipeline support with passing forced parameters

# %%
import argparse

parser = argparse.ArgumentParser(
    description="Process input and output paths for video processing."
)

parser.add_argument(
    "-i", "--input", 
    help="Path to the input video file.", 
    required=False
)

parser.add_argument(
    "-o", "--output", 
    help="Path to the output folder.", 
    required=False
)

parser.add_argument(
    "--f", "--kernel_launcher", 
    help="Path to the kernel launcher file.",
    required=False
)

# %%
args = parser.parse_args()

if args.input and args.output:
    args = parser.parse_args()
    
    FORCE_INPUT_AND_OUTPUT_PATHS = True
    FORCED_video_file_path = args.input
    FORCED_output_folder = args.output
    print("System arguments detected:", args)
else:
    FORCE_INPUT_AND_OUTPUT_PATHS = False


# %% [markdown]
# ### Helper

# %%
import re
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.cluster import DBSCAN
from pyproj import Transformer

# %%
def count_frames(video_path:str):
    """Count the number of frames in the video

    Args:
        video_path (str): The path to the video

    Returns:
        int: The number of frames in the video
    """
    
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return length

def get_file_and_output_folder(id:int):
    """Get the file path and output folder for the given id

    Args:
        id (int): The id of the video

    Returns:
        tuple[str, str]: The file path and the output_folder
    """
    
    file_path = f"{DATASET_PATH}/day/{id:03d}/R.MP4"
    output_folder = f"{DATASET_PATH}/day/{id:03d}/bends"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f'File {file_path} not found')
    
    print(f'Processing video {id:03d} with {count_frames(file_path)} frames')
    os.makedirs(output_folder, exist_ok=True)
    print(f'Results will be saved in {output_folder}')
    return file_path, output_folder

def time_stamp_to_seconds(time_stamp:str):
    """Convert a time stamp to seconds

    Args:
        time_stamp (str): The time stamp in the format HHMMSS:sss

    Returns:
        int: The seconds since midnight
    """
    
    hours = int(time_stamp[:2])
    minutes = int(time_stamp[2:4])
    seconds = int(time_stamp[4:6])
    milliseconds = int(time_stamp[7:])

    return (hours*60*60 + minutes*60 + seconds)*1000 + milliseconds

# %% [markdown]
# # Lat and Long to X and Y

# %%
# https://link-springer-com.manchester.idm.oclc.org/article/10.1007/s00190-023-01815-0
def lat_lon_to_x_y(lat:float, lon:float, height:float = 0):
    """Convert latitude, longitude and height to x, y and z

    Args:
        lat (float): latitude
        lon (float): longitude
        height (float): height above sea level
    """
    lat = np.radians(lat)
    lon = np.radians(lon)

    a = 6378137.0 # equatorial radius
    f = 0.003352810681183637418 # flattening

    e2 = f*(2-f) # first eccentricity squared

    Rn = a / np.sqrt(1 - e2*np.sin(lat)**2) # radius of curvature in the prime vertical

    lat = np.radians(lat)
    lon = np.radians(lon)

    # calculate x, y, z
    x = (Rn + height) * np.cos(lat) * np.cos(lon)
    y = (Rn + height) * np.cos(lat) * np.sin(lon)
    z = (Rn*(1-e2) + height) * np.sin(lat)

    return x, y, z

def lat_lon_to_BGS_X_Y(lat:float, lon:float):
    """Convert latitude and longitude to BGS X and Y

    Args:
        lat (float): latitude
        lon (float): longitude

    Returns:
        tuple: The BGS X and Y coordinates
    """
    # GPS - EPSG:4326  https://epsg.io/4326
    # BGS - EPSG:27700 https://epsg.io/27700
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700")
    bgs_format =  transformer.transform(lat, lon)

    northing = bgs_format[0]
    easting = bgs_format[1]

    return easting, northing

test_lat_kilburn = 53.4675254
test_lon_kilburn = -2.234003
test_lat_crewe = 53.09863295
test_lon_crewe = -2.45823645
test_conversion_kilburn = lat_lon_to_x_y(test_lat_kilburn, test_lon_kilburn)
test_conversion_crewe = lat_lon_to_x_y(test_lat_crewe, test_lon_crewe)
test_bsg_kilburn = lat_lon_to_BGS_X_Y(test_lat_kilburn, test_lon_kilburn)
test_bsg_crewe = lat_lon_to_BGS_X_Y(test_lat_crewe, test_lon_crewe)

print (f'Kilburn: X: {test_conversion_kilburn[0]}, Y: {test_conversion_kilburn[1]}')
print (f'Crewe: X: {test_conversion_crewe[0]}, Y: {test_conversion_crewe[1]}')
print (f'Kilburn BGS: X: {test_bsg_kilburn[0]}, Y: {test_bsg_kilburn[1]}')
print (f'Crewe BGS: X: {test_bsg_crewe[0]}, Y: {test_bsg_crewe[1]}')

conversion_distance_to_crewe = np.sqrt((test_conversion_kilburn[0] - test_conversion_crewe[0])**2 + (test_conversion_kilburn[1] - test_conversion_crewe[1])**2)
conversion_distance_to_crewe_bgs = np.sqrt((test_bsg_kilburn[0] - test_bsg_crewe[0])**2 + (test_bsg_kilburn[1] - test_bsg_crewe[1])**2)

print (f"According to google earth the distance between Kilburn and Crewe is 43,188.27 meters")
print(f'Distance between Kilburn and Crewe in meters: {conversion_distance_to_crewe}')
print(f'Distance between Kilburn and Crewe in meters (BGS): {conversion_distance_to_crewe_bgs}')

# %%
if FORCE_INPUT_AND_OUTPUT_PATHS:
    file_path = FORCED_video_file_path
    output_folder = FORCED_output_folder
else:
    file_path, output_folder = get_file_and_output_folder(ids[0])

# %%
# !firefox {file_path} # display the video

# %% [markdown]
# # Read file contents

# %% [markdown]
# ### NMEA string extraction

# %%
def getNMEAStringsFromFile(file_path:str):
    """Get the NMEA strings from the file

    Args:
        file_path (str): The path to the file

    Returns:
        List[str]: The GPRMC|GPGGA strings from the file
    """
    strings = ""
    with open(file_path, "rb") as f:
        strings = f.read()

    pattern =  rb'\$(?:GPRMC|GPGGA)[ -~]{' + str(min_string_length).encode() + rb',}'
    strings = re.findall(pattern, strings)    
    return [s.decode('utf-8', errors='ignore') for s in strings]

# %% [markdown]
# Get the NMEA strings from the video file.

# %%
data = getNMEAStringsFromFile(file_path)
frame_count = count_frames(file_path)
print(f'Found {len(data)} GPS records, of which {frame_count} frames are available')
data[:10]

# %% [markdown]
# NMEA string parsing and data extraction

# %%
def knots_to_mph(knots:float):
    """Convert knots to miles per hour

    Args:
        knots (float): The speed in knots

    Returns:
        float: The speed in miles per hour
    """
    return knots * 1.15078

def DMS_to_decimal(lon:str, lon_heading:str, lat:str, lat_heading:str):
    """Convert Degrees Minutes Seconds to decimal
    lon: dddmm.mmmm
    lat: ddmm.mmmm

    Args:
        lon (str): The longitude in DMS format
        lon_heading (str): The heading for the longitude
        lat (str): The latitude in DMS format
        lat_heading (str): The heading for the latitude

    Returns:
        float, float: The latitude and longitude in decimal format
    """
    
    lon_degrees = int(lon[:3])
    lon_minutes = float(lon[3:])
    lon_decimal = lon_degrees + lon_minutes/60
    if lon_heading == 'W':
        lon_decimal = -lon_decimal

    lat_degrees = int(lat[:2])
    lat_minutes = float(lat[2:])
    lat_decimal = lat_degrees + lat_minutes/60
    if lat_heading == 'S':
        lat_decimal = -lat_decimal

    return lat_decimal, lon_decimal

def parse_gprmc(input_sequence:str):
    """Parse the GPRMC and GPGGA string and extract the latitude, longitude, height and speed

    Args:
        input_sequence (str[]): [GPRMC string, GPGGA string]

    Returns:
        dict: The extracted values as a dictionary
    """
    # $GPRMC,<time>,<status>,<latitude>,<N/S>,<longitude>,<E/W>,<speed>,<course>,<date>,<magnetic variation>,<E/W>,<checksum>
    parts_GPRMC = input_sequence[0].split(',')
    if len(parts_GPRMC) < 10 or parts_GPRMC[0] != '$GPRMC':
        # print(f'Invalid GPRMC string: {input_sequence}')
        return None
    
    # $GPGGA,<time>,<latitude>,<N/S>,<longitude>,<E/W>,<quality>,<satellites>,<HDOP>,<height>,<height unit>,<geoid separation>,<geoid separation unit>,<age of differential data>,<station ID>,<checksum>
    parts_GPGGA = input_sequence[1].split(',')
    if len(parts_GPGGA) < 10 or parts_GPGGA[0] != '$GPGGA':
        # print(f'Invalid GPGGA string: {input_sequence}')
        return None

    
    # Extract latitude and longitude with direction
    time = parts_GPRMC[1]
    valid = parts_GPRMC[2]  # A - data valid, V - data invalid
    latitude = parts_GPRMC[3]
    lat_direction = parts_GPRMC[4]
    longitude = parts_GPRMC[5]
    lon_direction = parts_GPRMC[6]
    height = parts_GPGGA[9]  # Height above sea level
    speed = parts_GPRMC[7]  # Speed in knots

    # Convert latitude and longitude to decimal
    numeric_lat, numeric_lon = DMS_to_decimal(longitude, lon_direction, latitude, lat_direction)

    if valid == 'A': # check if the data record is valid
        speed = knots_to_mph(float(speed))  # Convert speed to mph
        
    # Return the extracted values as a dictionary
    return {
        "time": time,
        "valid": True if valid == 'A' else False,
        "latitude": numeric_lat,
        "longitude": numeric_lon,
        "height": height,
        "speed": speed
    }

# %%
# parse gps records at a step of two to capture both GPRMC and GPGGA records
positions = []
for i in range(0, len(data), 2):
    lat_lon = parse_gprmc(data[i:i+2])
    if lat_lon:
        positions.append(lat_lon)

# status of the records
print(f"Found {len(positions)} positions")
print("valid positions:", len([p for p in positions if p['valid']]))
print("invalid positions:", len([p for p in positions if not p['valid']]))

# remove records with the same values
positions = [dict(t) for t in {tuple(d.items()) for d in positions}]
print(f"Unique positions: {len(positions)}")

# %% [markdown]
# ### Video Metadata extraction

# %%
cap = cv2.VideoCapture(file_path, cv2.CAP_FFMPEG)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_rate = cap.get(cv2.CAP_PROP_FPS)

print(f"Total frames: {total_frames}")

print(f"Frame rate: {frame_rate}")

# %%
# filter out invalid GPS records
print(f"Total positions: {len(positions)}")
positions = [p for p in positions if p['valid']]
print(f"Valid positions: {len(positions)}")

positions[:2]

# %% [markdown]
# ### Convert Lat Lon to X, Y
# Convert to a 2D plane to convert out gps points to a meter based X, Y coordinate system
# - improves relative distance calculation accuracy
# - creates a relative scale between lon and lat
# - improves clustering accuracy
# - improves circle fitting accuracy

# %%
# process the positions to get x, y, z
# for position in positions:
#     lat, lon = position['latitude'], position['longitude']
#     lat = float(lat.split()[0])
#     lon = float(lon.split()[0])
#     h = float(position['height'])
#     position['x'], position['y'], position['z'] = lat_lon_to_x_y(lat, lon, h)

for position in positions:
    lat, lon = position['latitude'], position['longitude']
    position['x'], position['y'] = lat_lon_to_BGS_X_Y(lat, lon)


Original_Positions = positions.copy()
positions[:2]

# %% [markdown]
# # Plot GPS positions on graph

# %%
plt.close('all')
x = [pos['x'] for pos in positions]
y = [pos['y'] for pos in positions]

speeds = [pos['speed'] for pos in positions]

norm = plt.Normalize(min(speeds), max(speeds))
cmap = plt.cm.plasma

plt.scatter(x, y, c=speeds, cmap=cmap, norm=norm, alpha=0.5, s=1)
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Valid GPS positions > {min_speed_filter} MPH - Including GPS Outliers')

# %%
plt.close('all')

x = [pos['x'] for pos in positions]
y = [pos['y'] for pos in positions]

speeds = [pos['speed'] for pos in positions]

norm = plt.Normalize(min(speeds), max(speeds))
cmap = plt.cm.plasma

plt.scatter(x, y, c=speeds, cmap=cmap, norm=norm, alpha=0.5, s=1)
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Valid GPS positions > {min_speed_filter} MPH - Including GPS Outliers')

# %% [markdown]
# ### Remove outliers from our data

# %%
plt.close('all')

MIN_DISTANCE_OUTLIER_FILTER = 1000 # 1KM distance from the median

x = [pos['x'] for pos in positions]
y = [pos['y'] for pos in positions]

# calculate the median x and y
median_x = np.median(x)
median_y = np.median(y)
print(f'Median x: {median_x}')
print(f'Median y: {median_y}')

# filter out the points that are not within the MIN_DISTANCE_OUTLIER_FILTER
positions = [
    pos for pos in positions
        if abs(pos['x'] - median_x) < MIN_DISTANCE_OUTLIER_FILTER and 
            abs(pos['y'] - median_y) < MIN_DISTANCE_OUTLIER_FILTER
        ]

# plot the filtered points
x = [pos['x'] for pos in positions]
y = [pos['y'] for pos in positions]
speeds = [pos['speed'] for pos in positions]

plt.scatter(x, y, c=speeds, cmap=cmap, norm=norm, alpha=0.5, s=1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Valid GPS positions > {min_speed_filter} MPH')

# %%
# Get start and end time for frame estimation
positions = sorted(positions, key=lambda x: x['time'])
STARTING_TIME = time_stamp_to_seconds(positions[0]['time'])
print(f"Starting time: {STARTING_TIME}")

END_TIME = time_stamp_to_seconds(positions[-1]['time'])
print(f"Ending time: {END_TIME}")

# %% [markdown]
# ### Remove invalid records and records under min_speed_filter mph

# %%
def filter_invalid_and_low_mph_records(positions):
    """Filter the data points to remove
        - invalid points
        - points less than min_speed_filter mph
        - points with the same latitude and longitude
    
    Args:
        positions (List[dict]): The list of positions
    
    Returns:
        List[dict]: The filtered list of positions
    """
    
    filtered_positions = []
    for i in range(len(positions)):
        if not positions[i]['valid']:
            continue

        if positions[i]['speed'] < min_speed_filter:
            continue

        filtered_positions.append(positions[i])
    return filtered_positions

# %%
positions = filter_invalid_and_low_mph_records(positions)
positions = sorted(positions, key=lambda x: x['time']) # sort by time

print(f"{len(positions)} positions after filtering")
positions[:2]

# %% [markdown]
# # Find closest position entry to point of interest

# %%
def get_points_near_a_cluster_estimated_center(
        cluster_center:list[float],
        positions:list[dict],
        distance_threshold:float = 100):
    """Get the points near a cluster estimated center

    Args:
        cluster_center (list[float]): list of estimated centers
        positions (list[dict]): list of gps positions
        distance_threshold (float, optional): The distance threshold. Defaults to 100.

    Returns:
        list[dict]: The positions within the distance threshold
    """
    points = []
    for position in positions:
        x = float(position['x'])
        y = float(position['y'])
        distance = np.sqrt((x - cluster_center[0])**2 + (y - cluster_center[1])**2)
        if distance < distance_threshold:
            points.append(position)
    return points

# %% [markdown]
# # Find bends

# %%
def apply_median_filter(records:list[float], size:int = 5):
    """Apply a median filter to a list

    Args:
        records (list[float]): The list of points
        size (int, optional): Kernel size. Defaults to 5.

    Returns:
        list[float]: The filtered points
    """

    temp = records.copy() # create a copy to avoid modifying the original list
    # slide a window of size 2*size+1 over the records
    for i in range(size, len(records)-size):
        temp[i] = np.median(records[i-size:i+size])
    return temp

# %%
def get_accumulated_distance(x_pos: list[float], y_pos: list[float]) -> list[float]:
    """Get the accumulated distance from x and y positions

    Args:
        x_pos (list[float]): x values
        y_pos (list[float]): y values

    Returns:
        list[float]: list of accumulated distances
    """

    distance = 0.0
    distances = [0.0]
    for i in range(1, len(x_pos)):
        dx = x_pos[i] - x_pos[i-1]
        dy = y_pos[i] - y_pos[i-1]
        distance += np.hypot(dx, dy)
        distances.append(distance)
    return distances

def get_smoothed_sequence_angles(x_pos:list[float], y_pos:list[float], meters:int = 5) -> list[float]:
    """Get the smoothed sequence angles from x and y positions sequence

    Args:
        x_pos (list[float]): x positions
        y_pos (list[float]): y positions

    Returns:
        list[float]: The smoothed sequence angles
    """

    x_pos = apply_median_filter(x_pos, 3)
    y_pos = apply_median_filter(y_pos, 3)

    diff_x = np.diff(x_pos)
    diff_y = np.diff(y_pos)

    accumulated_distance = get_accumulated_distance(x_pos, y_pos)

    angles = []

    for i in range(1, len(accumulated_distance)):
        # find all positions within 5 meters before and after the current position
        current_distance = accumulated_distance[i]
        low_distance = max(0, current_distance - meters)
        high_distance = min(accumulated_distance[-1], current_distance + meters)

        low_index = np.searchsorted(accumulated_distance, low_distance, side='left')
        high_index = np.searchsorted(accumulated_distance, high_distance, side='right')
        high_index = min(high_index, len(accumulated_distance))

        # print(low_index, i, high_index)

        # Before vector (from low_index to i)
        before_vector = np.array([
            np.sum(diff_x[low_index:i]), 
            np.sum(diff_y[low_index:i])
        ])
        

        # After vector (from i to high_index)
        after_vector = np.array([
            np.sum(diff_x[i:high_index]), 
            np.sum(diff_y[i:high_index])
        ])
        
        # Check zero vectors
        if np.linalg.norm(before_vector) == 0 or np.linalg.norm(after_vector) == 0:
            angles.append(0.0)
            continue
        
        # Compute angle using vector operations
        dot = np.dot(before_vector, after_vector)
        cross = np.cross(before_vector, after_vector)  # Cross product magnitude (z-component)
        angle = np.arctan2(cross, dot)
        
        angles.append(angle)

    angles_rad = np.array(angles)
    complex_angles = np.exp(1j * angles_rad)

    # smooth angles with gaussian filter
    gaussian = np.exp(-np.linspace(-2, 2, 5)**2)
    gaussian /= gaussian.sum()
    smoothed_complex = np.convolve(complex_angles, gaussian, mode='same')

    # Convert back to angles
    smoothed_angles = np.angle(smoothed_complex)
    smoothed_degrees = np.degrees(smoothed_angles)

    return smoothed_degrees

# %% [markdown]
# ### Config parameters for bend ROI findings

# %%
lower_threshold = 5  # degrees
DB_SCAN_EPS = 10
DB_SCAN_MIN_SAMPLES = 3

# %%
plt.close('all')

positions_ordered_by_time = sorted(positions, key=lambda x: time_stamp_to_seconds(x['time']))

x_pos = [pos['x'] for pos in positions_ordered_by_time]
y_pos = [pos['y'] for pos in positions_ordered_by_time]
speeds = [pos['speed'] for pos in positions_ordered_by_time]

angles = get_smoothed_sequence_angles(x_pos, y_pos, meters=10)

print(f"Found {len(angles)} angles")
print(angles[:5])

# angles_derivative = np.gradient(angles) # get the first derivative of the angles

angles_derivative = angles # get the first derivative of the angles

plot_angles = angles_derivative
plt.title(f"Angle change by frame ({len(angles)} angles)")
plt.xlabel("Position")
plt.ylabel("Angle change")
plt.plot(plot_angles)
plt.savefig(os.path.join(output_folder, "angle_graph.png"))

# %%
plt.close('all')

# bends above the threshold
bends = np.where(np.abs(angles_derivative) > lower_threshold)

print(f"Found {len(bends)} bends")
print(angles_derivative[bends][:5])

# cluster bends
bends_positions = np.array(list(zip(x_pos, y_pos)))[bends]

temp_store_bends = bends_positions.copy() # Store the potential bends for plotting later

# bends = cluster.vq.kmeans(bends_positions, )[0] # not applicable since we need to know K
# Apply DBSCAN clustering (automatic K)
if len(bends_positions) > 1:
    dbscan = DBSCAN(eps=DB_SCAN_EPS, min_samples=DB_SCAN_MIN_SAMPLES, metric="euclidean")  # Adjust eps based on your GPS resolution
    labels = dbscan.fit_predict(bends_positions)
    
    # Get unique cluster centers
    unique_labels = set(labels)
    cluster_centers = np.array([bends_positions[labels == i].mean(axis=0) for i in unique_labels if i != -1])  # Ignore noise
else:
    cluster_centers = bends_positions  # If no clustering is needed, keep original bends

print(f"Found {len(cluster_centers)} bends after clustering")

plt.scatter(
    x_pos,
    y_pos,
    c=[cmap(norm(float(s))) for s in speeds],
    s=1,
    label='GPS Positions'
)

# add start and end points TEXT LABEL
plt.text(
    x_pos[0],
    y_pos[0],
    'Start',
    fontsize=9,
    color='black'
)

plt.text(
    x_pos[-1],
    y_pos[-1],
    'End',
    fontsize=9,
    color='black'
)

plt.scatter(
    temp_store_bends[:, 0],
    temp_store_bends[:, 1],
    color='black',
    label='Potential Bend',
    s=7
)

if len(cluster_centers) > 0:
    plt.scatter(
        cluster_centers[:,0],
        cluster_centers[:,1],
        color='red',
        label='Clustered Bends',
        s=40
    )
else:
    plt.legend(["GPS Positions", "Potential Bend"])

# set size
plt.gcf().set_size_inches(10, 10)
plt.title(f"All potential bends in GPS Path > {min_speed_filter} MPH")
plt.ylabel("y")
plt.xlabel("x")
plt.legend()
plt.axis('equal')
plt.gca().invert_xaxis()


plt.savefig(os.path.join(output_folder, "bends.png"))

# %%
print(f"total of {len(cluster_centers)} potential bend clusters found")
cluster_centers

# %% [markdown]
# ### Estimate relevant frames

# %%
def time_stamp_to_frame_number(time_stamp:str, STARTING_TIME:int = STARTING_TIME, END_TIME:int = END_TIME, total_frames:int = total_frames):
    """Estimate frame number using the timestamp

    Args:
        time_stamp (str): The time stamp to convert
        STARTING_TIME (int, optional): video starting time stamp. Defaults to STARTING_TIME.
        END_TIME (int, optional): video ending timestamp. Defaults to END_TIME.
        total_frames (int, optional): total number of frames. Defaults to total_frames.

    Returns:
        int: frame number
    """
    # calculate the difference between the time stamp and the starting time
    diff = time_stamp_to_seconds(time_stamp) - STARTING_TIME

    total_time = END_TIME - STARTING_TIME

    # print(f"diff: {diff}, toal_time: {total_time}")

    if total_time <= 0:
        return 0
    
    # estimate the frame number
    predict_frame_number = int((diff / total_time) * total_frames)
    
    return predict_frame_number

def get_closest_position_based_on_lat_lon(x:float, y:float, positions:list[dict]):
    """Get the closest position based on x and y

    Args:
        lat (float): x
        lon (float): y
        positions (list[dict], optional): list of all positions. Defaults to positions.

    Returns:
        dict: The closest position
    """
    min_distance = float('inf')
    closest_position = None

    # print(positions[:5])

    for position in positions:
        p_x = position['x']
        p_y = position['y']
        distance = np.sqrt((p_x - x)**2 + (p_y - y)**2)

        if distance < min_distance:
            min_distance = distance
            closest_position = position

    return closest_position


def get_frame_number_based_on_bend_x_y(cluster_centers:list[list[float]], positions:list[dict] = positions):
    """Get the frame number based on the bend x and y for each cluster center

    Args:
        cluster_centers (list[list[float]]): list of cluster centers
        positions (list[dict], optional): list of considered positions. Defaults to positions.

    Returns:
        list[int]: list of frame numbers
    """
    frame_numbers = []
    for bend in cluster_centers:
        closest_position = get_closest_position_based_on_lat_lon(bend[0], bend[1], positions)
        frame_number = time_stamp_to_frame_number(closest_position['time'])
        # print(f"frame_number: {frame_number}")
        frame_numbers.append(frame_number)

    return frame_numbers

# %%
frame_numbers = get_frame_number_based_on_bend_x_y(cluster_centers)

print(f"Estimated frame numbers: {frame_numbers}")

# %% [markdown]
# ### Output images of estimated frames

# %%

frame_numbers = sorted(frame_numbers)

def print_frames(frame_numbers:list[int], file_path:str = file_path):
    plt.close('all')
    
    frames = []

    cap = cv2.VideoCapture(file_path, cv2.CAP_FFMPEG)

    for frame_number in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            print(f"Warning: Could not read frame {frame_number}")
    
    cap.release()
    
    if frames:
        fig, axes = plt.subplots(1, len(frames), figsize=(20, 10))
        if len(frames) == 1:
            axes.imshow(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
            axes.axis('off')
            axes.set_title(f"Frame {frame_numbers[0]}")
        else:
            for i, frame in enumerate(frames):
                axes[i].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                axes[i].axis('off')
                axes[i].set_title(f"Frame {frame_numbers[i]}")
    else:
        print("No frames to display")

    return frames

frames = print_frames(frame_numbers)

print(f"Total frames: {total_frames}")

# %%
for i, frame in enumerate(frames):
    output_file = os.path.join(output_folder, f"bend_{i}.jpg")
    cv2.imwrite(output_file, frame)


# %% [markdown]
# # Calculate Bends in ROI

# %%
def plot_bend(bend_name:str, cluster_center:list[float], points:list[dict], focused_positions:list[dict]=None):
    """Plot the bend with the given name and points

    Args:
        bend_name (str): plot title
        points (list[dict]): list of points
    """
    plt.close('all')
    points = sorted(points, key=lambda x: time_stamp_to_seconds(x['time']))

    x = [float(pos['x']) for pos in points]
    y = [float(pos['y']) for pos in points]
    
    plt.scatter(x, y, s=1)
    plt.scatter(cluster_center[0], cluster_center[1], color='red', s=40)
    if not focused_positions is None:
        focused_x = [float(pos['x']) for pos in focused_positions]
        focused_y = [float(pos['y']) for pos in focused_positions]
        plt.scatter(focused_x, focused_y, color='green', s=40)

    plt.text(
        x[0],
        y[0],
        'Segment Start',
        fontsize=9,
        color='black'
    )

    plt.text(
        x[-1],
        y[-1],
        'Segment End',
        fontsize=9,
        color='black'
    )

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'GPS positions near {bend_name}')
    # flip x axis
    plt.gca().invert_xaxis()
    plt.axis('equal')

    plt.savefig(os.path.join(output_folder, f"{bend_name}.png"))

    if args.f:
        plt.show()
    else:
        plt.close('all')

# %% [markdown]
# find first curve of a series of points

# %%
from enum import auto


def find_first_bend_from_series(points:list[dict], min_degree_threshold:float = 2):
    """Find the initial bend from the series of points

    Args:
        points (list[dict]): list of points

    Returns:
        list[float]: The initial bend
    """
    points = sorted(points, key=lambda x: time_stamp_to_seconds(x['time']))

    x = [pos['x'] for pos in points]
    y = [pos['y'] for pos in points]

    # x = np.convolve(x, np.ones(3) /3, mode='valid')
    # y = np.convolve(y, np.ones(3) / 3, mode='valid')

    angles = get_smoothed_sequence_angles(x, y, meters=5)

    print(angles[:5])

    # set threshold to largest 10% of angles
    abs_angles = np.abs(angles)
    abs_angles = np.sort(abs_angles)
    auto_degree_threshold = abs_angles[int(len(abs_angles) * 0.85)]

    print("automatic threshold:", auto_degree_threshold)

    min_degree_threshold = max(min_degree_threshold, auto_degree_threshold)
    

    first_bend_found = False
    first_bend_sign = None

    tolerance = 3

    first_bend_positions = []
    for i in range(10, len(angles)):
        sign = np.sign(angles[i])
        if abs(angles[i]) > min_degree_threshold:
            if not first_bend_found:
                first_bend_positions.append(points[i])
                first_bend_found = True
                first_bend_sign = sign
            elif first_bend_sign == sign:
                first_bend_positions.append(points[i])
            
            continue

        if first_bend_found and first_bend_sign != sign:
            if tolerance <= 0:
                break
            tolerance -= 1  

    if not first_bend_found:
        print("No first bend found")
        return None, None

    # determine if the first bend is a left or right bend
    if first_bend_sign < 0:
        print("First bend is a left bend")
    else:
        print("First bend is a right bend")

    return first_bend_positions, first_bend_sign

# %%
bends_for_curve_fitting = []

for i, bend in enumerate(cluster_centers):
    records = get_points_near_a_cluster_estimated_center(bend, positions, distance_threshold=100)
    avg_speed = np.mean([float(pos['speed']) for pos in records])
    print(f"Bend {i}: {len(records)} points near the center - Avg Speed: {avg_speed} MPH")

    focused_points, first_bend_sign = find_first_bend_from_series(records, min_degree_threshold=5)

    if focused_points is not None:
        start_of_focused_points = focused_points[0]
        start_of_focused_points_frame = time_stamp_to_frame_number(start_of_focused_points['time'])
        print(f"Start of focused points: {start_of_focused_points_frame}")

    plot_bend(f"Bend {i}", bend, records, focused_points)

    if len(records) > 0 and focused_points is not None:
        bends_for_curve_fitting.append({
            "bend": bend,
            "focused_points": focused_points,
            "Estimated_start_frame": start_of_focused_points_frame,
            "first_bend_sign": "Left" if first_bend_sign < 0 else "Right",
            "avg_speed": avg_speed
        })

# %% [markdown]
# # Kasa Iterative Curve Fitting

# %%
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import least_squares

# class Circle:
#     def __init__(self, a, b, r, s, j):
#         self.a = a  # X-coordinate of the center
#         self.b = b  # Y-coordinate of the center
#         self.r = r  # Radius
#         self.s = s  # Root mean square error
#         self.j = j  # Total number of iterations

# class Data:
#     def __init__(self, X, Y):
#         self.X = np.array(X)
#         self.Y = np.array(Y)
#         self.n = len(X)

# def geometric_distance(params, X, Y):
#     a, b, r = params
#     return np.sqrt((X - a)**2 + (Y - b)**2) - r

# def Sigma(data, circle):
#     # Compute the root mean square error (estimate of sigma)
#     distances = np.sqrt((data.X - circle.a)**2 + (data.Y - circle.b)**2)
#     sigma = np.sqrt(np.mean((distances - circle.r)**2))
#     return sigma

# def CircleFitIterative(data, initial_guess=None, max_iterations=100, tol=1e-9):
#     if initial_guess is None:
#         # Use the Kasa method to obtain an initial guess
#         initial_guess = CircleFitByKasa(data)
#         initial_params = [initial_guess.a, initial_guess.b, initial_guess.r]
#     else:
#         initial_params = initial_guess

#     # Perform the iterative fitting using Levenberg–Marquardt algorithm
#     result = least_squares(geometric_distance, initial_params, args=(data.X, data.Y), max_nfev=max_iterations, ftol=tol, xtol=tol, gtol=tol)

#     # Extract the optimized parameters
#     a, b, r = result.x
#     s = Sigma(data, Circle(a, b, r, 0, 0))
#     j = result.nfev  # Number of function evaluations

#     return Circle(a, b, r, s, j)

# def CircleFitByKasa(data):
#     # Center the data points
#     Xi = data.X - np.mean(data.X)
#     Yi = data.Y - np.mean(data.Y)
#     Zi = Xi**2 + Yi**2

#     # Compute moments
#     Mxx = np.mean(Xi**2)
#     Myy = np.mean(Yi**2)
#     Mxy = np.mean(Xi * Yi)
#     Mxz = np.mean(Xi * Zi)
#     Myz = np.mean(Yi * Zi)

#     # Solve the system of equations using Cholesky factorization
#     G11 = np.sqrt(Mxx)
#     G12 = Mxy / G11
#     G22 = np.sqrt(Myy - G12**2)

#     D1 = Mxz / G11
#     D2 = (Myz - D1 * G12) / G22

#     # Compute parameters of the fitting circle
#     C = D2 / G22 / 2
#     B = (D1 - G12 * C) / G11 / 2

#     # Assemble the output
#     a = B + np.mean(data.X)
#     b = C + np.mean(data.Y)
#     r = np.sqrt(B**2 + C**2 + Mxx + Myy)
#     s = Sigma(data, Circle(a, b, r, 0, 0))
#     j = 0  # Number of iterations (not applicable for Kasa's method)

#     return Circle(a, b, r, s, j)

# f_bend = bends_for_curve_fitting[1]['points']
# order_by_time = sorted(f_bend, key=lambda x: time_stamp_to_seconds(x['time']))
# X = [float(pos['x']) for pos in order_by_time]
# Y = [float(pos['y']) for pos in order_by_time]

# diff_x = np.diff(X)
# diff_y = np.diff(Y)
# angles = np.arctan2(diff_x, diff_y)
# angles = np.arctan2(np.sin(angles), np.cos(angles)) # normalize angles
# angles = np.degrees(angles)

# angles = np.diff(angles)

# # smooth angles with gaussian filter
# angles = np.convolve(angles, np.ones(3) / 3, mode='same')


# # apply median filter
# angles = apply_median_filter(angles, 10)

# # plot angles
# plt.plot(angles)
# plt.show()

# min_degree = 0.5

# print(f"Found {len(X)} points before filtering")
# print(angles[:10])

# angle_threshold_mask = [np.abs(angles) > min_degree]
# print(angle_threshold_mask[:10])

# X_a = np.array(X[1:-1])[np.abs(angles) > min_degree]
# Y_a = np.array(Y[1:-1])[np.abs(angles) > min_degree]

# # # ignore after sign change
# # angles_temp = angles[np.abs(angles) > min_degree]
# # angle_sign = np.sign(angles_temp)
# # initial_angle = angle_sign[1]

# # X_a = []
# # Y_a = []

# # for i in range(1, min(len(X)-1, len(angles))):
# #     if np.abs(angles[i-1]) > min_degree:
# #         if not (angle_sign[i-1] != initial_angle):
# #             break
# #         X_a.append(X[i])
# #         Y_a.append(Y[i])


# print(f"Found {len(X_a)} points after filtering")

# data = Data(X_a,Y_a)
# fitted_circle = CircleFitIterative(data)


# print(f"Center: ({fitted_circle.a}, {fitted_circle.b})")
# print(f"Radius: {fitted_circle.r}")
# print(f"RMS Error: {fitted_circle.s}")
# print(f"Iterations: {fitted_circle.j}")

# plt.scatter(X, Y)
# plt.scatter(fitted_circle.a, fitted_circle.b, color='red')
# plt.scatter(X_a, Y_a, color='green')
# plt.text(
#     X[0],
#     Y[0],
#     'Segment Start',
#     fontsize=9,
#     color='black'
# )

# plt.text(
#     X[-1],
#     Y[-1],
#     'Segment End',
#     fontsize=9,
#     color='black'
# )

# circle = plt.Circle((fitted_circle.a, fitted_circle.b), fitted_circle.r, color='r', fill=False)

# plt.gca().add_artist(circle)
# plt.axis('equal')
# plt.show()


# %% [markdown]
# # Visualise

# %%
if bends_for_curve_fitting:
	frames = print_frames([bend['Estimated_start_frame'] for bend in bends_for_curve_fitting])
	for i, frame in enumerate(frames):
		output_file = os.path.join(output_folder, f"bend_{i}_frame_{bends_for_curve_fitting[i]['Estimated_start_frame']}_{bends_for_curve_fitting[i]['first_bend_sign']}.jpg")
		cv2.imwrite(output_file, frame)
else:
	print("No bends found for curve fitting.")
	


# %% [markdown]
# # Output Bend CSV

# %%
columns = ["frame", "bend_direction", "avg_speed"]
data = []

for bend in bends_for_curve_fitting:
    data.append([bend['Estimated_start_frame'], bend['first_bend_sign'],bend['avg_speed']])

import pandas as pd
df = pd.DataFrame(data, columns=columns)
df

# %%
csv_save_path = os.path.join(output_folder, "bend_directions.csv")
df.to_csv(csv_save_path, index=False)
print(f"Saved bend directions to {csv_save_path}")

# %%



