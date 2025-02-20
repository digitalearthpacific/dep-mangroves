from logging import INFO, Formatter, Logger, StreamHandler, getLogger

import boto3
import numpy as np
import typer
import xarray as xr
from dask.distributed import Client
from dep_tools.aws import object_exists
from dep_tools.exceptions import EmptyCollectionError
from dep_tools.grids import PACIFIC_GRID_10
from dep_tools.loaders import OdcLoader
from dep_tools.namers import S3ItemPath
from dep_tools.processors import Processor
from dep_tools.searchers import PystacSearcher
from dep_tools.stac_utils import StacCreator
from dep_tools.task import AwsStacTask as Task
from dep_tools.writers import AwsDsCogWriter
from odc.geo import Geometry
from odc.stac import configure_s3_access
from typing_extensions import Annotated
from utils import get_gmw
from xarray import DataArray

OUTPUT_NODATA = -32767


def get_logger(region_code: str, name: str) -> Logger:
    """Set up a simple logger"""
    console = StreamHandler()
    time_format = "%Y-%m-%d %H:%M:%S"
    console.setFormatter(
        Formatter(
            fmt=f"%(asctime)s %(levelname)s ({region_code}):  %(message)s",
            datefmt=time_format,
        )
    )

    log = getLogger(name)
    log.addHandler(console)
    log.setLevel(INFO)
    return log


class MangrovesProcessor(Processor):
    def __init__(self, areas: Geometry, scale: float = 10_000, offset: float = 0):
        super().__init__()
        self.areas = areas
        self.scale = scale
        self.offset = offset

    def process(self, data: DataArray) -> DataArray:
        data = data.squeeze()

        # Scale and offset the data
        data = (data * (1 / self.scale) + self.offset).clip(0, 1)

        # Mask to only keep areas identified as mangroves in the GMW dataset
        data = data.odc.mask(self.areas)

        # Create NDVI
        data["ndvi"] = (data.nir - data.red) / (data.nir + data.red)

        # Create an empty DataArray to store the mangroves classification
        data["mangroves"] = xr.full_like(data.ndvi, OUTPUT_NODATA, dtype="int16")

        # Classify so that less than 0.4 is 0, between 0.4 and 0.7 is 1, and greater than 0.7 is 2
        data["mangroves"] = xr.where(data.ndvi <= 0.4, 0, data.mangroves)
        data["mangroves"] = xr.where(
            (data.ndvi > 0.4) & (data.ndvi <= 0.7), 1, data.mangroves
        )
        data["mangroves"] = xr.where(
            (data.ndvi > 0.7), 2, data.mangroves
        )

        # Mask nodata from the NDVI
        data["mangroves"] = data.mangroves.where(data.ndvi.notnull(), OUTPUT_NODATA)

        # Only keep the mangroves band
        data = data[["mangroves"]]

        return data


def main(
    tile_id: Annotated[str, typer.Option()],
    datetime: Annotated[str, typer.Option()],
    version: Annotated[str, typer.Option()],
    output_bucket: str = None,
    base_product: str = "s2",
    memory_limit: str = "50GB",
    n_workers: int = 2,
    threads_per_worker: int = 32,
    scale: Annotated[float | None, typer.Option()] = 10_000,
    offset: Annotated[float | None, typer.Option()] = 0,
    decimated: bool = False,
    overwrite: Annotated[bool, typer.Option()] = False,
) -> None:
    log = get_logger(tile_id, "MANGROVES")
    log.info("Starting processing.")

    grid = PACIFIC_GRID_10

    tile_index = tuple(int(i) for i in tile_id.split(","))
    geobox = grid.tile_geobox(tile_index)

    if decimated:
        log.warning("Running at 1/10th resolution")
        geobox = geobox.zoom_out(10)

    gmw = get_gmw()
    geom = geobox.geographic_extent.to_crs(gmw.crs)

    areas = gmw.intersection(geom)

    # Make sure we can access S3
    log.info("Configuring S3 access")
    configure_s3_access(cloud_defaults=True)

    client = boto3.client("s3")

    itempath = S3ItemPath(
        bucket=output_bucket,
        sensor=base_product,
        dataset_id="mangroves",
        version=version,
        time=datetime,
    )
    stac_document = itempath.stac_path(tile_id)

    # If we don't want to overwrite, and the destination file already exists, skip it
    if not overwrite and object_exists(output_bucket, stac_document, client=client):
        log.info(f"Item already exists at {stac_document}")
        # This is an exit with success
        raise typer.Exit()

    catalog = "https://stac.digitalearthpacific.org"
    collection = "dep_s2_geomad"

    searcher = PystacSearcher(
        catalog=catalog, collections=[collection], datetime=datetime
    )

    loader = OdcLoader(
        bands=["red", "nir"],
        # chunks=[-1, 2048, 2048],
        groupby="solar_day",
        fail_on_error=False,
        clip_to_area=False,
    )

    processor = MangrovesProcessor(areas=areas, scale=scale, offset=offset)

    # Custom writer so we write multithreaded
    writer = AwsDsCogWriter(itempath, write_multithreaded=True)

    # STAC making thing
    stac_creator = StacCreator(
        itempath=itempath, remote=True, make_hrefs_https=True, with_raster=True
    )

    try:
        with Client(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit,
        ):
            log.info(
                (
                    f"Started dask client with {n_workers} workers "
                    f"and {threads_per_worker} threads with "
                    f"{memory_limit} memory"
                )
            )
            paths = Task(
                itempath=itempath,
                id=tile_index,
                area=geobox,
                searcher=searcher,
                loader=loader,
                processor=processor,
                writer=writer,
                logger=log,
                stac_creator=stac_creator,
            ).run()
    except EmptyCollectionError:
        log.info("No items found for this tile")
        raise typer.Exit()  # Exit with success
    except Exception as e:
        log.exception(f"Failed to process with error: {e}")
        raise typer.Exit(code=1)

    log.info(
        f"Completed processing. Wrote {len(paths)} items to https://{output_bucket}.s3.us-west-2.amazonaws.com/{ stac_document}"
    )


if __name__ == "__main__":
    typer.run(main)
