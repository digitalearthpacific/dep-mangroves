import geopandas as gpd


full_grid = (
    gpd.read_file(
        "https://raw.githubusercontent.com/digitalearthpacific/dep-grid/master/grid_pacific.geojson"
    )
    .astype({"code": str, "gid": str})
    .set_index(["code", "gid"], drop=False)
)

# This was downloaded from the official site. When I get a sec I'll put
# the origin url here
url = "https://deppcpublicstorage.blob.core.windows.net/input/gmw/gmw_v3_2020_vec.zip"
gmw_full = gpd.read_file(url)

gmw_clip = gmw_full.to_crs(full_grid.crs).clip(full_grid)

grid_gmw = full_grid.intersection(gmw_clip.unary_union)

grid_gmw = (
    grid_gmw[~grid_gmw.is_empty]
    .to_frame("geometry")
    .join(full_grid[["country", "code", "gid"]])
)

gmw_clip.to_parquet("data/gmw_v3_2020_vec_dep.parquet")
gmw_clip.to_file("data/gmw_v3_2020_vec_dep.gpkg")

grid_gmw.to_parquet("data/grid_gmw_v3_2020_vec.parquet")
