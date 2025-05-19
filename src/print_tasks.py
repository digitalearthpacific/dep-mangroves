import json
import sys
from typing import Annotated, Generator, Optional

import boto3
import typer
from dep_tools.aws import object_exists
from dep_tools.grids import PACIFIC_EPSG, PACIFIC_GRID_10, gadm
from dep_tools.namers import S3ItemPath
from odc.geo.geom import Geometry
from shapely import STRtree

from utils import get_gmw


def get_tasks(
    country_codes: list[str | None], years: list, version: str
) -> Generator[dict, None, None]:
    # Get GMW data
    gmw = get_gmw()
    gmw_index = STRtree([g.geom for g in gmw])

    # If we have country codes, filter the tiles by the countries
    countries = None
    gadm_all = gadm()
    if country_codes is not None:
        countries = gadm_all.loc[lambda df: df["GID_0"].isin(country_codes)]
    else:
        countries = gadm_all
    countries = Geometry(countries.union_all(), "epsg:4326")

    # Set up a big list of tiles, all in memory
    tiles = PACIFIC_GRID_10.tiles(countries.to_crs(PACIFIC_EPSG).boundingbox)
    tiles = [t for t in tiles]  # Change to a list

    # This is an optimised intersection, using a spatial index
    # to find the tiles that intersect with the GMW data
    # and then filtering them by the countries
    # if we have them
    aoi_tiles = []
    for tile in tiles:
        projected = tile[1].geographic_extent.to_crs(PACIFIC_EPSG)

        if gmw_index.query(projected.geom).any():
            aoi_tiles.append(tile)

    # Finally, punch out a JSON list of tasks
    for tile in aoi_tiles:
        if countries is not None:
            geometry = tile[1].geographic_extent
            # Check if the tile intersects with the countries
            if not countries.intersects(geometry):
                # It doesn't, so skip it
                continue
        # If we get here, the tile intersects with the GMW data
        # and the countries, so we can add it to the list of tasks
        # for each year.
        for year in years:
            yield {
                "tile-id": ",".join([str(i) for i in tile[0]]),
                "year": year,
                "version": version,
            }


def main(
    years: Annotated[str, typer.Option()],
    version: Annotated[str, typer.Option()],
    regions: Optional[str] = "ALL",
    limit: Optional[str] = None,
    output_bucket: Optional[str] = None,
    output_prefix: Optional[str] = None,
    overwrite: Annotated[bool, typer.Option()] = False,
) -> None:
    country_codes = None if regions.upper() == "ALL" else regions.split(",")

    if limit is not None:
        limit = int(limit)

    # Makes a list no matter what
    years = years.split("-")
    if len(years) == 2:
        years = range(int(years[0]), int(years[1]) + 1)
    elif len(years) > 2:
        ValueError(f"{years} is not a valid value for --years")

    tasks = get_tasks(country_codes, years, version)

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
