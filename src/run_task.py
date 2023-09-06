import ast
from typing_extensions import Annotated

import geopandas as gpd
import typer
from xarray import DataArray
import xrspatial.multispectral as ms
from xrspatial.classify import reclassify
import numpy as np
import rioxarray as rx

from azure_logger import CsvLogger
from dep_tools.loaders import Sentinel2OdcLoader
from dep_tools.namers import DepItemPath
from dep_tools.processors import Processor
from dep_tools.runner import run_by_area_dask_local
from dep_tools.stac_utils import set_stac_properties
from dep_tools.utils import get_container_client
from dep_tools.writers import AzureDsWriter

from grid import grid


class MangrovesProcessor(Processor):
    def process(self, xr: DataArray) -> DataArray:
        median = xr.median("time")
        ds = (
            ms.ndvi(median.sel(band="B08"), median.sel(band="B04")).to_dataset(
                name="ndvi"
            )
            # I usually eschew compute here but as the writer writes individual
            # variables as separate tif tiles data are re-pulled for each dervied
            # dataset below. If we get memory errors then we could remove this,
            # things will take a bit longer though (and there will be a lot more
            # network traffic).
            .compute()
        )
        ds["mangroves"] = reclassify(ds.ndvi, [0.4, np.inf], [float("nan"), 1])
        ds["regular"] = reclassify(
            ds.ndvi, [0.4, 0.7, np.inf], [float("nan"), 1, float("nan")]
        )
        ds["closed"] = reclassify(ds.ndvi, [0.7, np.inf], [float("nan"), 1])

        return set_stac_properties(xr, ds)


def get_gmw_shapes_for_area(area: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return (
        gpd.read_file(
            "data/gmw_v3_2020_vec.shp",
            mask=area,
        )
        .dissolve("PXLVAL")
        .set_index(area.index)
    )


def main(
    region_code: Annotated[str, typer.Option()],
    region_index: Annotated[str, typer.Option()],
    datetime: Annotated[str, typer.Option()],
    version: Annotated[str, typer.Option()],
    local_cluster_kwargs: Annotated[str, typer.Option(..., callback=ast.literal_eval)],
    dataset_id: str = "mangroves",
) -> None:
    cell = grid.loc[[(region_code, region_index)]]

    area = get_gmw_shapes_for_area(cell)

    loader = Sentinel2OdcLoader(
        epsg=3832,
        datetime=datetime,
        dask_chunksize=dict(band=1, time=1, x=4096, y=4096),
        odc_load_kwargs=dict(fail_on_error=False, resolution=10, bands=["B04", "B08"]),
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
        areas=area,
        loader=loader,
        processor=processor,
        writer=writer,
        logger=logger,
        continue_on_error=False,
        local_cluster_kwargs=local_cluster_kwargs,
    )


if __name__ == "__main__":
    typer.run(main)
