import os
import requests
from haversine import haversine, Unit

# OneMap API endpoints
TOKEN_URL = "https://www.onemap.gov.sg/api/auth/post/getToken"
SEARCH_URL = "https://www.onemap.gov.sg/api/common/elastic/search"


# Authenticate with OneMap and return API token
def authenticate():
    email = os.getenv("ONEMAP_EMAIL")
    password = os.getenv("ONEMAP_PASSWORD")

    if not email or not password:
        raise ValueError("OneMap credentials not set in environment variables.")

    payload = {"email": email, "password": password}
    response = requests.post(TOKEN_URL, json=payload)
    response.raise_for_status()

    return response.json().get("access_token")


# Get Coordinates of a Given Address with OneMap API
def get_lat_long(address, api_token):
    try:
        response = requests.get(
            SEARCH_URL,
            params={"searchVal": address, "returnGeom": "Y", "getAddrDetails": "N"},
            headers={"Authorization": api_token}
        )
        response.raise_for_status()
        results = response.json().get("results", [])

        if results:
            return float(results[0]["LATITUDE"]), float(results[0]["LONGITUDE"])

        return None, None

    except Exception as e:
        print(f"Failed for {address}: {e}")
        return None, None


# Get Nearest Location and Distance from Reference DataFrame
def find_nearest(lat, lon, reference_df):
    # Compute Haversine Distance to Every Row
    distances = reference_df.apply(
        lambda row: haversine(
            (lat, lon),
            (row["Lat"], row["Long"]),
            unit=Unit.KILOMETERS
        ),
        axis=1
    )
    min_index = distances.idxmin()

    return (
        reference_df.loc[min_index, "Address"],
        round(distances[min_index], 3)
    )


# Compute nearest MRT and Mall for a Given Address
def compute_nearest_for_address(address, mrt_df, mall_df):
    # Authenticate and get token
    token = authenticate()

    # Geocode address
    lat, lon = get_lat_long(address, token)
    if lat is None or lon is None:
        return None

    # Find nearest MRT
    nearest_mrt, mrt_distance = find_nearest(lat, lon, mrt_df)

    # Find nearest Mall
    nearest_mall, mall_distance = find_nearest(lat, lon, mall_df)

    return {
        "Nearest_MRT": nearest_mrt,
        "Distance_to_MRT": mrt_distance,
        "Nearest_Mall": nearest_mall,
        "Distance_to_Mall": mall_distance
    }