from pathlib import Path

import geopandas as gpd
from dep_tools.grids import PACIFIC_EPSG
from odc.geo import Geometry
from xarray import DataArray, full_like, where

OUTPUT_NODATA = 255


def get_gmw_parquet() -> gpd.GeoSeries:
    current_dir = Path(__file__).parent
    gmw_file = current_dir / "gmw_pacific_new.parquet"

    # Open as a file handle to avoid pyarrow's LocalFileSystem() construction,
    # which conflicts with GDAL's bundled libarrow on macOS.
    # See https://github.com/apache/arrow/issues/44696
    with open(gmw_file, "rb") as f:
        return gpd.read_parquet(f)


def get_gmw() -> Geometry:
    gmw = Geometry(get_gmw_parquet().to_geo_dict(), crs=PACIFIC_EPSG)

    return gmw


def process_mangroves(
    data: DataArray, areas: Geometry, scale: float = 0.0001, offset: float = 0.0
) -> DataArray:
    data = data.squeeze()

    # Scale and offset the data
    data = (data * scale + offset).clip(0, 1)

    # Mask to only keep areas identified as mangroves in the GMW dataset
    data = data.odc.mask(areas)

    # Create NDVI
    data["ndvi"] = (data.nir - data.red) / (data.nir + data.red)

    # Create an empty DataArray to store the mangroves classification
    data["mangroves"] = full_like(data.ndvi, OUTPUT_NODATA, dtype="uint8")

    # Classify so that less than 0.4 is 0, between 0.4 and 0.7 is 1, and greater than 0.7 is 2
    data["mangroves"] = where(data.ndvi <= 0.4, 0, data.mangroves)
    data["mangroves"] = where((data.ndvi > 0.4) & (data.ndvi <= 0.7), 1, data.mangroves)
    data["mangroves"] = where((data.ndvi > 0.7), 2, data.mangroves)

    # Mask nodata from the NDVI
    data["mangroves"] = data.mangroves.where(data.ndvi.notnull(), OUTPUT_NODATA)

    # Only keep the mangroves band and set nodata
    data = data[["mangroves"]].astype("uint8")
    data.mangroves.odc.nodata = OUTPUT_NODATA

    return data
