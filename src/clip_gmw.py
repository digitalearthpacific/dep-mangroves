import geopandas as gpd


from grid import grid

# This was downloaded from the official site. When I get a sec I'll put
# the origin url here
url = "https://deppcpublicstorage.blob.core.windows.net/input/gmw/gmw_v3_2020_vec.zip"
gmw_full = gpd.read_file(url)

gmw_clip = gmw_full.to_crs(grid.crs).clip(grid)

gmw_clip.to_parquet("data/gmw_v3_2020_vec_dep.parquet")
gmw_clip.to_file("data/gmw_v3_2020_vec_dep.gpkg")
