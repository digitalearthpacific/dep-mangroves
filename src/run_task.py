import numpy as np
import rioxarray as rx
import typer
import xrspatial.multispectral as ms
from azure_logger import CsvLogger
from dep_tools.loaders import Sentinel2OdcLoader
from dep_tools.namers import DepItemPath
from dep_tools.processors import Processor
from dep_tools.runner import run_by_area_dask_local
from dep_tools.utils import get_container_client
from dep_tools.writers import AzureDsWriter
from rasterio.warp import transform_bounds
from typing_extensions import Annotated
from xarray import DataArray

from grid import grid


class MangrovesProcessor(Processor):
    def process(self, xr: DataArray) -> DataArray:
        median = xr.resample(time="1Y").median().squeeze()
        gmw = load_gmw(xr)
        ndvi = (
            ms.ndvi(median.sel(band="B08"), median.sel(band="B04"))
            .where(gmw)
            .to_dataset(name="mangrove")
        )
        masked = ndvi.where(gmw.squeeze())
        mangroves = xr.where(masked > 0.4, 1, np.nan)
        regular_mangroves = mangroves.where(masked <= 0.7)
        closed_mangroves = mangroves.where(masked > 0.7)
        regular_mangroves = regular_mangroves.rename({"mangrove": "regular"})
        closed_mangroves = closed_mangroves.rename({"mangrove": "closed"})
        mangroves = xr.merge([mangroves, regular_mangroves, closed_mangroves])
        mangroves = mangroves.squeeze()
        return mangroves


def load_gmw(ds) -> DataArray:
    input_path = "https://deppcpublicstorage.blob.core.windows.net/input/gmw/gmw_v3_2020_ras_dep.tif"

    gmw = rx.open_rasterio(input_path, chunks=True)
    bounds = list(transform_bounds(ds.rio.crs, gmw.rio.crs, *ds.rio.bounds()))
    return gmw.rio.clip_box(*bounds).squeeze().rio.reproject_match(ds)


def main(
    region_code: Annotated[str, typer.Option()],
    region_index: Annotated[str, typer.Option()],
    datetime: Annotated[str, typer.Option()],
    version: Annotated[str, typer.Option()],
    dataset_id: str = "mangroves",
) -> None:
    cell = grid.loc[[(region_code, region_index)]]

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
        path=itempath.log_path(),
        overwrite=False,
        header="time|index|status|paths|comment\n",
    )

    run_by_area_dask_local(
        areas=cell,
        loader=loader,
        processor=processor,
        writer=writer,
        logger=logger,
        continue_on_error=False,
    )


if __name__ == "__main__":
    typer.run(main)
