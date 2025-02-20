import json
import sys
from typing import Annotated, Generator, Optional

import boto3
import typer
from dep_tools.aws import object_exists
from dep_tools.grids import PACIFIC_EPSG, get_tiles
from dep_tools.namers import S3ItemPath

from utils import get_gmw


def get_tasks(tiles: list, years: list, version: str) -> Generator[dict, None, None]:
    gmw = get_gmw()

    for year in years:
        for tile in tiles:
            tile_geom = tile[1].geographic_extent.to_crs(PACIFIC_EPSG)

            # Check if the tile intersects with the gmw
            if tile_geom.intersects(gmw):
                yield {
                    "tile-id": ",".join([str(i) for i in tile[0]]),
                    "year": year,
                    "version": version,
                }


def main(
    years: Annotated[str, typer.Option()],
    version: Annotated[str, typer.Option()],
    regions: Optional[str] = "ALL",
    tile_buffer_kms: Optional[int] = 0.0,
    limit: Optional[str] = None,
    output_bucket: Optional[str] = None,
    output_prefix: Optional[str] = None,
    overwrite: Annotated[bool, typer.Option()] = False,
) -> None:
    country_codes = None if regions.upper() == "ALL" else regions.split(",")

    tiles = get_tiles(
        country_codes=country_codes, buffer_distance=tile_buffer_kms * 1000
    )

    if limit is not None:
        limit = int(limit)

    # Makes a list no matter what
    years = years.split("-")
    if len(years) == 2:
        years = range(int(years[0]), int(years[1]) + 1)
    elif len(years) > 2:
        ValueError(f"{years} is not a valid value for --years")

    tasks = get_tasks(tiles, years, version)

    # If we don't want to overwrite, then we should only run tasks that don't already exist
    # i.e., they failed in the past or they're missing for some other reason
    if not overwrite:
        valid_tasks = []
        client = boto3.client("s3")
        for task in tasks:
            itempath = S3ItemPath(
                bucket=output_bucket,
                sensor="s2",
                dataset_id="geomad",
                version=version,
                time=task["year"],
            )
            stac_path = itempath.stac_path(task["tile-id"].split(","))

            if output_prefix is not None:
                stac_path = f"{output_prefix}/{stac_path}"

            if not object_exists(output_bucket, stac_path, client=client):
                valid_tasks.append(task)

            # Save time if we have a limit
            if len(valid_tasks) == limit:
                break
        # Switch to this list of tasks, which has been filtered
        tasks = valid_tasks
    else:
        # If we are overwriting, we just keep going, making them a list not a generator
        tasks = [t for t in tasks]

    if limit is not None:
        tasks = tasks[0:limit]

    json.dump(tasks, sys.stdout)


if __name__ == "__main__":
    typer.run(main)
