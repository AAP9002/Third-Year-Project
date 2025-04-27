import os
import sys
import pandas as pd
import folium
from folium.plugins import MarkerCluster

def main(csv_path, output_html="map.html"):
    # 1. Load data
    if not os.path.isfile(csv_path):
        print(f"Error: file not found: {csv_path}")
        sys.exit(1)
    df = pd.read_csv(csv_path)

    # 2. Basic validation
    if not {'latitude', 'longitude'}.issubset(df.columns):
        print("Error: CSV must contain 'latitude' and 'longitude' columns.")
        sys.exit(1)

    # 3. Center map on the mean location
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()

    # 4. Create the map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=2)

    # 5. Add a MarkerCluster
    # marker_cluster = MarkerCluster(name="Points").add_to(m)

    # 6. Add points to the cluster
    for idx, row in df.iterrows():
        # folium.Marker(
        #     location=(row['latitude'], row['longitude']),
        #     popup=str(idx)  # or any other info you want to show
        # ).add_to(marker_cluster)
        folium.CircleMarker(
            location=(row['latitude'], row['longitude']),
            radius=3,
            color=None,
            fill=True,
            fill_color='blue',
            fill_opacity=0.7,
            popup=str(idx)
        ).add_to(m)  # add straight to the map, not to a cluster

    # 7. Save to HTML
    m.save(output_html)
    print(f"Map with {len(df)} points saved to: {output_html}")

if __name__ == "__main__":
    csv_file = 'merged_long_lats.csv'
    out_file = "map.html"
    main(csv_file, out_file)
