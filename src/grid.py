import fsspec
import geopandas as gpd

grid_url = "https://deppcpublicstorage.blob.core.windows.net/input/gmw/grid_gmw_v3_2020_vec.parquet"

with fsspec.open(grid_url) as file:
    grid = gpd.read_parquet(file)
