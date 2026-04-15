import os, glob, math
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from pyproj import CRS, Transformer

STATIONS_DIR = Path("stations")
OUT_DIR = Path("kelowna_regions2")
OUT_DIR.mkdir(exist_ok=True)

OUTPUT_GEOJSON = OUT_DIR / "kelowna_36_regions.geojson"
OUTPUT_SHP = OUT_DIR / "kelowna_36_regions.shp"

# Detect lat/lon columns
LAT_COLS = ["lat","latitude","Latitude","LAT"]
LON_COLS = ["lon","lng","longitude","Longitude","LON"]

def find_lat_lon(df):
    lat = next((c for c in df.columns if c in LAT_COLS), None)
    lon = next((c for c in df.columns if c in LON_COLS), None)
    if not lat or not lon:
        raise Exception("Could not detect lat/lon columns")
    return lat, lon

# Load all station coords
lats, lons = [], []
for f in glob.glob(str(STATIONS_DIR/"fine_area*_stations.csv")):
    df = pd.read_csv(f)
    latc, lonc = find_lat_lon(df)
    lats += df[latc].tolist()
    lons += df[lonc].tolist()

# Choose UTM zone (Kelowna ~ zone 11N)
zone = int((sum(lons)/len(lons) + 180) / 6) + 1
utm = CRS.from_epsg(32600 + zone)

to_utm = Transformer.from_crs("EPSG:4326", utm, always_xy=True).transform
to_wgs = Transformer.from_crs(utm, "EPSG:4326", always_xy=True).transform

# Convert to UTM
xy = [to_utm(lon, lat) for lon, lat in zip(lons, lats)]
xs = [p[0] for p in xy]; ys = [p[1] for p in xy]

minx, maxx = min(xs), max(xs)
miny, maxy = min(ys), max(ys)

width = maxx - minx
height = maxy - miny
side = max(width, height)  # make a perfect square

# Adjust bbox to square
cx = (minx + maxx)/2; cy = (miny + maxy)/2
half = side/2
minx, maxx = cx-half, cx+half
miny, maxy = cy-half, cy+half

# Build 6x6 grid
regions = []
nx, ny = 6, 6
dx = (maxx - minx) / nx
dy = (maxy - miny) / ny

region_id = 1
for i in range(ny):
    for j in range(nx):
        x1 = minx + j*dx; x2 = x1 + dx
        y1 = maxy - (i+1)*dy; y2 = y1 + dy

        poly = Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)])
        poly_wgs = Polygon([to_wgs(x,y) for x,y in poly.exterior.coords])

        regions.append({"region_id": region_id, "geometry": poly_wgs})
        region_id += 1

# Save
gdf = gpd.GeoDataFrame(regions, crs="EPSG:4326")
gdf.to_file(OUTPUT_GEOJSON, driver="GeoJSON")
gdf.to_file(OUTPUT_SHP)

print("Uniform 6x6 grid created")
print("Saved:", OUTPUT_GEOJSON, OUTPUT_SHP)
