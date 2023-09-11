import ast
import sys
import warnings

import numpy as np
import typer
import xrspatial.multispectral as ms
from azure_logger import CsvLogger, filter_by_log
from dep_tools.loaders import Sentinel2OdcLoader
from dep_tools.namers import DepItemPath
from dep_tools.processors import Processor
from dep_tools.runner import run_by_area_dask_local
from dep_tools.s2_utils import scale_and_offset_s2
from dep_tools.stac_utils import set_stac_properties
from dep_tools.utils import get_container_client
from dep_tools.writers import AzureDsWriter
from typing_extensions import Annotated
from xarray import DataArray
from xrspatial.classify import reclassify


MANGROVES_BASE_PRODUCT = "s2"
MANGROVES_DATASET_ID = "mangroves"
output_nodata = -32767


class MangrovesProcessor(Processor):
    def process(self, xr: DataArray) -> DataArray:
        xr = scale_and_offset_s2(xr)
        # TODO: consider cloud mask here
        # Also, think about keeping data as INT as long as possible
        # to reduce memory usage. The S2Loader converts to Float...

        # Load into memory
        median = xr.median("time").compute()

        ds = ms.ndvi(median.sel(band="B08"), median.sel(band="B04")).to_dataset(
            name="ndvi"
        )

        ds["mangroves"] = reclassify(ds.ndvi, [0.4, np.inf], [float("nan"), 1]).astype(
            int
        )

        ds["regular"] = reclassify(
            ds.ndvi, [0.4, 0.7, np.inf], [float("nan"), 1, float("nan")]
        ).astype(int)

        ds["closed"] = reclassify(ds.ndvi, [0.7, np.inf], [float("nan"), 1]).astype(int)

        return set_stac_properties(xr, ds)


def main(
    datetime: Annotated[str, typer.Option()],
    version: Annotated[str, typer.Option()],
    region_code: Annotated[str, typer.Option()] = "",
    region_index: Annotated[str, typer.Option()] = "",
    local_cluster_kwargs: Annotated[str, typer.Option()] = "",
    dataset_id: str = MANGROVES_DATASET_ID,
) -> None:
    from grid import grid

    areas = grid

    # None would be better for default but typer doesn't support it (str|None)
    if region_code != "":
        areas = grid[grid.index.get_level_values("code").isin([region_code])]

    if region_index != "":
        areas = grid[grid.index == (region_code, region_index)]

    if len(areas) == 0:
        warnings.warn(
            f"index ({region_code}, {region_index}) not found in grid, no output produced"
        )
        sys.exit(0)

    loader = Sentinel2OdcLoader(
        epsg=3832,
        datetime=datetime,
        dask_chunksize=dict(band=1, time=1, x=4096, y=4096),
        odc_load_kwargs=dict(fail_on_error=False, resolution=10, bands=["B04", "B08"]),
    )

    processor = MangrovesProcessor()
    itempath = DepItemPath(MANGROVES_BASE_PRODUCT, dataset_id, version, datetime)

    writer = AzureDsWriter(
        itempath=itempath,
        overwrite=False,
        output_nodata=output_nodata,
        extra_attrs=dict(dep_version=version),
    )

    logger = CsvLogger(
        name=dataset_id,
        container_client=get_container_client(),
        path=itempath.log_path(),
        overwrite=False,
        header="time|index|status|paths|comment\n",
    )

    local_cluster_kwargs_dict = (
        ast.literal_eval(local_cluster_kwargs) if local_cluster_kwargs != "" else dict()
    )
    areas = filter_by_log(areas, logger.parse_log())
    run_by_area_dask_local(
        areas=areas,
        loader=loader,
        processor=processor,
        writer=writer,
        logger=logger,
        continue_on_error=False,
        local_cluster_kwargs=local_cluster_kwargs_dict,
    )


if __name__ == "__main__":
    typer.run(main)
