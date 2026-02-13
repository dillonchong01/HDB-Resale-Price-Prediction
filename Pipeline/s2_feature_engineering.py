import pandas as pd
from Pipeline.utils import authenticate, get_lat_long, find_nearest

# Dataset file paths
HDB_FEATURES = r"Datasets\HDB_Features.csv"
MRT_COORDS = r"Datasets\MRT_LatLong.csv"
MALL_COORDS = r"Datasets\Mall_LatLong.csv"


# Ensure coordinate files (MRT / Mall) have complete Lat/Long
def update_reference_coordinates(file_path, api_token):
    print(f"\nChecking coordinates for {file_path}...")

    df = pd.read_csv(file_path)
    update_count = 0

    for index, row in df.iterrows():

        # If coordinates are missing, fetch from OneMap
        if pd.isna(row["Lat"]) or pd.isna(row["Long"]):

            lat, lon = get_lat_long(row["Address"], api_token)
            if lat is None:
                continue

            df.at[index, "Lat"] = lat
            df.at[index, "Long"] = lon
            update_count += 1

            # Save checkpoint every 50 updates
            if update_count % 50 == 0:
                df.to_csv(file_path, index=False)
                print(f"Checkpoint saved at {update_count} updates.")

    df.to_csv(file_path, index=False)
    print(f"Finished updating {file_path}.")


# Update HDB_Features with nearest MRT and Mall distances
def update_hdb_features(api_token):
    print("\nProcessing HDB_Features.csv...")

    df = pd.read_csv(HDB_FEATURES)
    mrt_df = pd.read_csv(MRT_COORDS)
    mall_df = pd.read_csv(MALL_COORDS)

    update_count = 0

    for index, row in df.iterrows():

        # Skip rows without address
        if pd.isna(row["Address"]):
            continue

        lat = row["Lat"]
        lon = row["Long"]
        row_updated = False

        # Fetch coordinates if missing
        if pd.isna(lat) or pd.isna(lon):

            lat, lon = get_lat_long(row["Address"], api_token)
            if lat is None:
                continue

            df.at[index, "Lat"] = lat
            df.at[index, "Long"] = lon
            row_updated = True

        # Compute nearest MRT if missing
        if pd.isna(row.get("Distance_to_MRT")):

            nearest_mrt, mrt_distance = find_nearest(lat, lon, mrt_df)

            df.at[index, "Nearest_MRT"] = nearest_mrt
            df.at[index, "Distance_to_MRT"] = mrt_distance
            row_updated = True

        # Compute nearest Mall if missing
        if pd.isna(row.get("Distance_to_Mall")):

            nearest_mall, mall_distance = find_nearest(lat, lon, mall_df)

            df.at[index, "Nearest_Mall"] = nearest_mall
            df.at[index, "Distance_to_Mall"] = mall_distance
            row_updated = True

        if row_updated:
            update_count += 1

        # Save checkpoint every 25 updates
        if update_count % 25 == 0:
            df.to_csv(HDB_FEATURES, index=False)
            print(f"Checkpoint saved at {update_count} updates.")

    df.to_csv(HDB_FEATURES, index=False)
    print("Finished updating HDB_Features.csv.")


if __name__ == "__main__":

    # Authenticate with OneMap
    token = authenticate()

    if token is None:
        print("Authentication failed.")
        exit(1)

    # Update coordinate reference files first
    update_reference_coordinates(MRT_COORDS, token)
    update_reference_coordinates(MALL_COORDS, token)

    # Then update HDB features
    update_hdb_features(token)