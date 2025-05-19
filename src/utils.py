from pathlib import Path

import geopandas as gpd
from dep_tools.grids import PACIFIC_EPSG
from odc.geo import Geometry


def get_gmw_parquet() -> gpd.GeoSeries:
    current_dir = Path(__file__).parent
    gmw_file = current_dir / "gmw_pacific_new.parquet"

    return gpd.read_parquet(gmw_file)


def get_gmw() -> Geometry:
    gmw = Geometry(get_gmw_parquet().to_geo_dict(), crs=PACIFIC_EPSG)

    return gmw
