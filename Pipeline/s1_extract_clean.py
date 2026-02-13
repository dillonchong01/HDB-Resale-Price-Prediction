import pandas as pd
import numpy as np
import re

# File Paths
RAW_DF = r"Datasets\Resale_Data.csv"
HDB_FEATURES = r"Datasets\HDB_Features.csv"
CLEANED_DF = r"Datasets\Cleaned_Resale_Data.csv"

if __name__ == "__main__":
    # Load Main Dataset
    df = pd.read_csv(RAW_DF)
    hdb_coordinates = pd.read_csv(HDB_FEATURES)

    # Extract Year and Month
    df["month"] = pd.to_datetime(df["month"])
    df["Year"] = df["month"].dt.year
    df["Month"] = df["month"].dt.month

    # Get Address
    df['Address'] = df['block'].astype(str) + " " + df['street_name'].astype(str)

    # Extract Floor Level
    df["Floor Level"] = df["storey_range"].str.split(" TO ").str[0].astype(int)

    # Convert Remaining Lease to Decimal
    lease_extract = df["remaining_lease"].str.extract(
        r'(?P<years>\d+)\s*years?(?:\s*(?P<months>\d+)\s*months?)?'
    )
    lease_extract = lease_extract.fillna(0).astype(int)
    df["Remaining Lease"] = lease_extract["years"] + lease_extract["months"] / 12

    # Rename Columns
    df = df.rename(columns={
        "town": "Town",
        "flat_type": "Flat Type",
        "floor_area_sqm": "Floor Area",
        "resale_price": "Resale Price",
        "lease_commence_date": "Lease Commence Date"
    })

    # Add Mature Column
    MATURE_ESTATES = [
        "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT MERAH", "BUKIT TIMAH", "CENTRAL", "CLEMENTI",
        "GEYLANG", "KALLANG/WHAMPOA", "MARINE PARADE", "PASIR RIS", "QUEENSTOWN", "SERANGOON",
        "TAMPINES", "TOA PAYOH"
    ]
    df["Mature"] = df["Town"].isin(MATURE_ESTATES).astype(int)

    # Add New Addresses to HDB Coordinates
    new_rows = df.loc[~df["Address"].isin(hdb_coordinates["Address"])]

    if not new_rows.empty:
        # Aggregate Town, Mature, Lease Commence Date by mode
        new_rows_agg = new_rows.groupby("Address", as_index=False).agg({
            "Town": "first",
            "Mature": "first",
            "Lease Commence Date": "first"
        })

        # Create Floor Area Map for each Address
        floor_area_map = (
            new_rows
            .groupby(["Address", "Flat Type"])["Floor Area"]
            .agg(lambda x: x.mode().iloc[0])
            .unstack()
            .apply(lambda row: row.dropna().to_dict(), axis=1)
            .to_dict()
        )

        new_rows_agg["Floor Area Map"] = new_rows_agg["Address"].map(floor_area_map)

        # Add NaN features for coordinates/MRT/Mall
        new_rows_agg["Lat"] = np.nan
        new_rows_agg["Long"] = np.nan
        new_rows_agg["Nearest_MRT"] = np.nan
        new_rows_agg["Distance_to_MRT"] = np.nan
        new_rows_agg["Nearest_Mall"] = np.nan
        new_rows_agg["Distance_to_Mall"] = np.nan

        # Concatenate to HDB Coordinates
        hdb_coordinates = pd.concat([hdb_coordinates, new_rows_agg], ignore_index=True)

    # Drop Unused Columns
    df = df.drop(columns=["flat_model", "month", "block", "street_name", "storey_range",
                          "remaining_lease", "Lease Commence Date"])

    # Save Cleaned Data and HDB Features
    df.to_csv(CLEANED_DF, index=False)
    hdb_coordinates.to_csv(HDB_FEATURES, index=False)