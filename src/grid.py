import geopandas as gpd

grid = (
    gpd.read_file(
        "https://raw.githubusercontent.com/digitalearthpacific/dep-grid/master/grid_pacific.geojson"
    )
    .astype({"code": str, "gid": str})
    .set_index(["code", "gid"], drop=False)
)

# breakpoint()
# gmw = gpd.read_file("gmw_v3_2020_vec.shp")
# intersection = grid.intersection(gmw)
#
# grid.geometry = intersection
# output = grid[~grid.is_empty]
#
# breakpoint()
# output.to_file("data/coastline_split_by_pathrow.gpkg")
