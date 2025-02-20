from pathlib import Path

import geopandas as gpd
from dep_tools.grids import PACIFIC_EPSG
from odc.geo import Geometry


def get_gmw() -> Geometry:
    current_dir = Path(__file__).parent
    gmw_file = current_dir / "gmw_pacific.parquet"
    gmw = Geometry(gpd.read_parquet(gmw_file).to_geo_dict(), crs=PACIFIC_EPSG)

    return gmw
