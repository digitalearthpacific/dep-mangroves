from typing_extensions import Annotated
from typing import List

import geopandas as gpd
import typer
from rasterio.warp import transform_bounds
import rioxarray as rx
from xarray import DataArray, Dataset
import xrspatial.multispectral as ms

from azure_logger import CsvLogger, get_log_path
from dep_tools.loaders import Sentinel2OdcLoader
from dep_tools.namers import DepItemPath
from dep_tools.processors import Processor
from dep_tools.runner import run_by_area_dask
from dep_tools.utils import get_container_client
from dep_tools.writers import AzureDsWriter

from grid import grid


class MangrovesProcessor(Processor):
    def process(self, xr: DataArray) -> DataArray:
        median = xr.resample(time="1Y").median().squeeze()
        gmw = load_gmw(xr)
        return ms.ndvi(median.sel(band="B08"), median.sel(band="B04")).where(gmw)


def load_gmw(ds) -> DataArray:
    input_path = "https://deppcpublicstorage.blob.core.windows.net/input/gmw/gmw_v3_2020_ras_dep.tif"

    gmw = rx.open_rasterio(input_path, chunks=True)
    bounds = list(transform_bounds(ds.rio.crs, gmw.rio.crs, *ds.rio.bounds()))
    return (
        gmw.rio.clip_box(*bounds)
        .squeeze()
        .rio.reproject_match(ds)
        .to_dataset(name="ndvi")
    )


def main(
    # region_code: str,
    # region_index: str,
    datetime: str = "2022",
    version: str = "test.0.1",
    dataset_id: str = "mangroves",
) -> None:
    # cell = grid.loc[[(region_code, region_index)]]

    loader = Sentinel2OdcLoader(
        epsg=3832,
        datetime=datetime,
        dask_chunksize=dict(band=1, time=1, x=4096, y=4096),
        odc_load_kwargs=dict(
            fail_on_error=False,
            resolution=10,
        ),
    )

    processor = MangrovesProcessor()
    itempath = DepItemPath("s2", dataset_id, version, datetime)

    writer = AzureDsWriter(
        itempath=itempath,
        convert_to_int16=True,
        overwrite=True,
        output_value_multiplier=10000,
        extra_attrs=dict(dep_version=version),
        write_stac=True,
    )
    logger = CsvLogger(
        name=dataset_id,
        container_client=get_container_client(),
        path=get_log_path(dataset_id, version, datetime),
        overwrite=False,
        header="time|index|status|paths|comment\n",
    )

    run_by_area_dask(
        areas=grid,
        loader=loader,
        processor=processor,
        writer=writer,
        logger=logger,
        continue_on_error=True,
    )


if __name__ == "__main__":
    typer.run(main)
