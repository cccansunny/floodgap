"""
FloodGap - scoring pipeline
===========================
Scores every census tract in the study region for flood-sensor siting priority,
runs the five weighting schemes, and writes the robustness table.

Folder layout expected (run from the repository root):

    floodgap/
      code/floodgap_compound.py      <- this file
      data/
        Vulnerability/Texas.csv                  CDC SVI 2022 (Texas)
        Vulnerability/Georgia.csv                CDC SVI 2022 (Georgia)
        Vulnerability/2024_gaz_tracts_48.txt     Census gazetteer, Texas tracts
        Vulnerability/Georgia_census_gazatteer.txt
        Flood Risk/tide_*.csv                    NOAA sea-level-rise viewer exports
        Impervious/2021_CCAP_J1414090tR0_C0.tif  NOAA C-CAP 30 m land cover
        elev_cache_houston.csv                   cached USGS elevations (lat,lon,elevation_m)
        elev_cache_georgia.csv
      output/                                    created automatically

Run:   cd floodgap
       python code/floodgap_compound.py

Requires: pandas, numpy, rasterio, pyproj   (pip install pandas numpy rasterio pyproj)
requests is only needed if the elevation cache is missing.

Every step below is tagged with the five data-analysis actions:
READ -> CLEAN -> JOIN -> COMPUTE -> OUTPUT
"""

import os
import numpy as np
import pandas as pd

# ===========================================================================
# CONFIG
# ===========================================================================

REGION = "houston"          # "houston" or "georgia"

# Paths are relative to the repository root (the folder that contains data/).
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT  = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)

SVI_CSV = {
    "houston": os.path.join(DATA, "Vulnerability", "Texas.csv"),
    "georgia": os.path.join(DATA, "Vulnerability", "Georgia.csv"),
}[REGION]

GAZETTEER = {
    "houston": os.path.join(DATA, "Vulnerability", "2024_gaz_tracts_48.txt"),
    "georgia": os.path.join(DATA, "Vulnerability", "Georgia_census_gazatteer.txt"),
}[REGION]

ELEV_CACHE = os.path.join(DATA, f"elev_cache_{REGION}.csv")

CCAP_TIF = os.path.join(DATA, "Impervious", "2021_CCAP_J1414090tR0_C0.tif")

# Study counties. NOTE the word-boundary match below: a plain substring match
# for "Harris" also catches "Harrison County" (nowhere near Houston) - a bug
# that silently inflated the study area to 1,227 tracts before it was caught.
COUNTIES = {
    "houston": ["Harris", "Galveston"],
    "georgia": ["Chatham", "Glynn", "Camden", "Liberty", "McIntosh", "Bryan", "Effingham"],
}[REGION]

# NOAA stations: (name, lat, lon, observed high-tide flood days).
# The day counts are the values used throughout the semester, extracted from
# the NOAA sea-level-rise viewer exports kept in data/Flood Risk/.
NOAA_STATIONS = {
    "houston": [
        ("Morgans Point, TX",          29.6817, -94.9850, 10.667),
        ("Eagle Point, Galveston, TX", 29.4800, -94.9170, 34.0),
        ("Galveston Pier 21, TX",      29.3100, -94.7933, 18.0),
    ],
    "georgia": [
        ("Fort Pulaski, GA",           32.0367, -80.9017, 14.333),
        ("Fernandina Beach, FL",       30.6714, -81.4658,  9.333),
        ("Mayport, FL",                30.3935, -81.4300,  3.0),
    ],
}[REGION]

# C-CAP developed-class impervious coefficients (class value -> impervious
# fraction), from the NOAA C-CAP classification scheme (see ccap-scheme.pdf):
#   2 developed high intensity, 3 medium, 4 low, 5 developed open space.
CCAP_IMPERV = {2: 0.8503, 3: 0.5768, 4: 0.2929, 5: 0.10}
CCAP_WINDOW = 17     # raster half-window in pixels; 17 * 30 m ~ 0.5 km each side

# The 11 major bayous / creeks as simplified waypoint chains (lon, lat).
# Straight-line distance to the nearest densified point stands in for
# riverine exposure. Not hydraulic modeling - stated as a limitation.
BAYOUS = {
    "Buffalo":   [(-95.80, 29.78), (-95.60, 29.77), (-95.45, 29.76), (-95.37, 29.76), (-95.20, 29.75), (-95.08, 29.74)],
    "Brays":     [(-95.75, 29.65), (-95.55, 29.68), (-95.40, 29.70), (-95.28, 29.72)],
    "WhiteOak":  [(-95.55, 29.90), (-95.45, 29.84), (-95.40, 29.79), (-95.36, 29.77)],
    "Greens":    [(-95.55, 29.95), (-95.40, 29.92), (-95.33, 29.88), (-95.23, 29.78)],
    "Sims":      [(-95.55, 29.62), (-95.42, 29.64), (-95.35, 29.66), (-95.27, 29.70)],
    "Halls":     [(-95.45, 29.90), (-95.36, 29.86), (-95.30, 29.83)],
    "SanJac":    [(-95.30, 30.10), (-95.20, 30.00), (-95.15, 29.95), (-95.05, 29.80)],
    "Cypress":   [(-95.75, 29.98), (-95.60, 30.00), (-95.45, 30.00), (-95.25, 30.03)],
    "Clear":     [(-95.35, 29.52), (-95.25, 29.53), (-95.15, 29.55), (-95.02, 29.55)],
    "Dickinson": [(-95.15, 29.45), (-95.05, 29.45), (-94.95, 29.45)],
    "Chocolate": [(-95.35, 29.30), (-95.28, 29.25), (-95.20, 29.20)],
}

# The five weighting schemes. Weights are POLICY OPTIONS, not fitted values -
# see the semester report for why no validated "correct" weights exist.
#   sub = (tidal, storm, river) inside the compound flood
#   out = (flood, vulnerability) in the final score
SCHEMES = [
    ("A", "Tidal-only",         (1.0, 0.0, 0.0),      (0.60, 0.40)),
    ("B", "Equal sub-weights",  (1/3, 1/3, 1/3),      (0.60, 0.40)),
    ("C", "Harvey-informed",    (0.25, 0.50, 0.25),   (0.60, 0.40)),
    ("D", "Flood-dominant",     (1/3, 1/3, 1/3),      (0.80, 0.20)),
    ("E", "Equity-forward",     (1/3, 1/3, 1/3),      (0.50, 0.50)),
]


# ===========================================================================
# Helpers
# ===========================================================================

def minmax(s):
    """COMPUTE: scale a series to 0-1. Every factor goes through this before weighting."""
    s = pd.Series(s, dtype=float)
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else s * 0.0


def haversine_miles(lat1, lon1, lat2, lon2):
    """COMPUTE: great-circle distance in miles. Accepts arrays on either side."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


# ===========================================================================
# Step 1  READ + CLEAN: CDC SVI
# ===========================================================================

def load_svi():
    df = pd.read_csv(SVI_CSV)
    df["FIPS"] = df["FIPS"].astype(str).str.zfill(11)
    # Word-boundary county match (the Harrison-County fix).
    pattern = r"\b(?:" + "|".join(COUNTIES) + r")\b"
    df = df[df["COUNTY"].str.contains(pattern, case=False, na=False, regex=True)].copy()
    df = df[df["RPL_THEMES"] >= 0].copy()          # CLEAN: -999 = missing in SVI
    out = df[["FIPS", "COUNTY", "RPL_THEMES", "E_TOTPOP"]].rename(columns={
        "FIPS": "tract_id", "RPL_THEMES": "svi", "E_TOTPOP": "population"})
    print(f"Step 1  SVI: {len(out)} tracts in {COUNTIES}")
    return out


# ===========================================================================
# Step 2  JOIN (key merge): tract coordinates from the Census gazetteer
# ===========================================================================

def attach_coords(tracts):
    with open(GAZETTEER, "r", encoding="utf-8", errors="ignore") as fh:
        header = fh.readline()
    sep = "|" if header.count("|") > header.count("\t") else "\t"
    gaz = pd.read_csv(GAZETTEER, sep=sep, dtype=str)
    gaz.columns = [c.strip() for c in gaz.columns]
    gaz = gaz.rename(columns={"INTPTLAT": "lat", "INTPTLONG": "lon", "GEOID": "tract_id"})
    gaz["tract_id"] = gaz["tract_id"].astype(str).str.zfill(11)
    gaz["lat"] = pd.to_numeric(gaz["lat"], errors="coerce")
    gaz["lon"] = pd.to_numeric(gaz["lon"], errors="coerce")
    merged = tracts.merge(gaz[["tract_id", "lat", "lon"]], on="tract_id", how="left")
    merged = merged.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    print(f"Step 2  coords: {len(merged)} tracts with gazetteer coordinates")
    return merged


# ===========================================================================
# Step 3  JOIN (nearest distance): NOAA flood days from the nearest station
# ===========================================================================

def attach_flood_days(df):
    st_lat = np.array([s[1] for s in NOAA_STATIONS])
    st_lon = np.array([s[2] for s in NOAA_STATIONS])
    st_day = np.array([s[3] for s in NOAA_STATIONS])
    st_name = [s[0] for s in NOAA_STATIONS]
    days, names = [], []
    for _, r in df.iterrows():
        d = haversine_miles(r["lat"], r["lon"], st_lat, st_lon)
        i = int(np.argmin(d))
        days.append(st_day[i]); names.append(st_name[i])
    df["flood_days"] = days
    df["nearest_station"] = names
    print(f"Step 3  flood days attached; stations used: {sorted(set(names))}")
    return df


# ===========================================================================
# Step 4  JOIN (key merge) or READ (API): USGS elevation
# ===========================================================================

def attach_elevation(df):
    if os.path.exists(ELEV_CACHE):
        cache = pd.read_csv(ELEV_CACHE)
        cache["key"] = cache["lat"].round(4).astype(str) + "," + cache["lon"].round(4).astype(str)
        df["key"] = df["lat"].round(4).astype(str) + "," + df["lon"].round(4).astype(str)
        df = df.merge(cache[["key", "elevation_m"]], on="key", how="left").drop(columns="key")
        got = df["elevation_m"].notna().sum()
        print(f"Step 4  elevation: {got}/{len(df)} from cache {os.path.basename(ELEV_CACHE)}")
    else:
        df["elevation_m"] = np.nan
        print("Step 4  elevation cache missing - querying Open-Elevation API")
    missing = df["elevation_m"].isna()
    if missing.any():
        import requests, time
        pts = df.loc[missing, ["lat", "lon"]]
        vals = []
        for start in range(0, len(pts), 100):          # API accepts batches
            chunk = pts.iloc[start:start + 100]
            locs = "|".join(f"{a:.4f},{b:.4f}" for a, b in zip(chunk["lat"], chunk["lon"]))
            try:
                r = requests.get("https://api.open-elevation.com/api/v1/lookup",
                                 params={"locations": locs}, timeout=30)
                vals += [p["elevation"] for p in r.json()["results"]]
            except Exception as e:
                print(f"   API error ({e}); filling with median")
                vals += [np.nan] * len(chunk)
            time.sleep(1)
        df.loc[missing, "elevation_m"] = vals
        df["elevation_m"] = df["elevation_m"].fillna(df["elevation_m"].median())
        # OUTPUT: refresh the cache so the next run is offline
        df[["lat", "lon", "elevation_m"]].to_csv(ELEV_CACHE, index=False)
    return df


# ===========================================================================
# Step 5  JOIN (window sample): impervious fraction from the C-CAP raster
# ===========================================================================

def attach_impervious(df):
    """For each tract center, sample a ~1 km window of the 30 m C-CAP raster
    and average the developed-class impervious coefficients. Water/forest -> 0."""
    import rasterio
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    src = rasterio.open(CCAP_TIF)
    arr = src.read(1)
    H, W = arr.shape
    vals = []
    for _, r in df.iterrows():
        x, y = tr.transform(r["lon"], r["lat"])
        row, col = src.index(x, y)
        if 0 <= row < H and 0 <= col < W:
            win = arr[max(0, row - CCAP_WINDOW):row + CCAP_WINDOW + 1,
                      max(0, col - CCAP_WINDOW):col + CCAP_WINDOW + 1]
            v = win[win != 0]                            # CLEAN: 0 = nodata
            vals.append(float(np.mean([CCAP_IMPERV.get(int(z), 0.0) for z in v.flatten()]))
                        if len(v) else np.nan)
        else:
            vals.append(np.nan)
    df["impervious_frac"] = vals
    df["impervious_frac"] = df["impervious_frac"].fillna(df["impervious_frac"].median())
    print(f"Step 5  impervious: mean {df['impervious_frac'].mean():.2f}, "
          f"max {df['impervious_frac'].max():.2f}")
    return df


# ===========================================================================
# Step 6  JOIN (nearest distance): riverine = distance to the bayou network
# ===========================================================================

def attach_riverine(df):
    pts = []
    for wps in BAYOUS.values():                          # densify each segment
        for (x1, y1), (x2, y2) in zip(wps[:-1], wps[1:]):
            n = max(2, int(max(abs(x2 - x1), abs(y2 - y1)) / 0.005))
            for k in range(n + 1):
                pts.append((x1 + (x2 - x1) * k / n, y1 + (y2 - y1) * k / n))
    blon = np.array([p[0] for p in pts]); blat = np.array([p[1] for p in pts])
    df["river_dist_mi"] = [float(haversine_miles(r["lat"], r["lon"], blat, blon).min())
                           for _, r in df.iterrows()]
    print(f"Step 6  riverine: distance {df['river_dist_mi'].min():.1f}-"
          f"{df['river_dist_mi'].max():.1f} mi")
    return df


# ===========================================================================
# Step 7  COMPUTE: normalize factors, score the five schemes, robustness
# ===========================================================================

def compute_schemes(df):
    df["low_lying_norm"] = 1 - minmax(df["elevation_m"])
    df["svi_norm"] = minmax(df["svi"])
    df["tidal_c"] = 0.5 * minmax(df["flood_days"]) + 0.5 * df["low_lying_norm"]
    if REGION == "houston":
        df["storm_c"] = minmax(df["impervious_frac"])
        df["river_c"] = 1 - minmax(df["river_dist_mi"])
    else:                                                # Georgia: tidal only
        df["storm_c"] = 0.0
        df["river_c"] = 0.0

    for key, name, sub, out in SCHEMES:
        comp = minmax(sub[0] * df["tidal_c"] + sub[1] * df["storm_c"] + sub[2] * df["river_c"])
        df[f"sc_{key}"] = out[0] * comp + out[1] * df["svi_norm"]
        df[f"in50_{key}"] = (df[f"sc_{key}"].rank(ascending=False) <= 50).astype(int)

    df["n50"] = df[[f"in50_{k}" for k, *_ in SCHEMES]].sum(axis=1)
    df["mean_rank"] = df[[f"sc_{k}" for k, *_ in SCHEMES]].rank(ascending=False).mean(axis=1)

    print("\nStep 7  scheme results (Harris / Galveston in top 50):")
    for key, name, sub, out in SCHEMES:
        top = df.nlargest(50, f"sc_{key}")
        h = int(top["COUNTY"].str.contains("Harris").sum())
        print(f"   {key} {name:18s}: {h} / {50 - h}")
    print("Robustness tally:", {n: int((df['n50'] == n).sum()) for n in range(5, 0, -1)})
    return df


# ===========================================================================
# Step 8  OUTPUT
# ===========================================================================

def main():
    df = load_svi()
    df = attach_coords(df)
    df = attach_flood_days(df)
    df = attach_elevation(df)
    if REGION == "houston":
        df = attach_impervious(df)
        df = attach_riverine(df)
    df = compute_schemes(df)

    scored = os.path.join(OUT, f"floodgap_{REGION}_scored.csv")
    df.to_csv(scored, index=False)
    print(f"\nStep 8  wrote {scored}")

    if REGION == "houston":
        robust = os.path.join(OUT, "houston_robust.csv")
        df.to_csv(robust, index=False)
        top20 = df.sort_values(["n50", "mean_rank"], ascending=[False, True]).head(20)
        top20.to_csv(os.path.join(OUT, "houston_robust_top20.csv"), index=False)
        h = int(top20["COUNTY"].str.contains("Harris").sum())
        print(f"        robust top 20: {h} Harris / {20 - h} Galveston")


if __name__ == "__main__":
    main()
