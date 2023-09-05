from itertools import product
import os
from typing import List

# to silence warning
os.environ["USE_PYGEOS"] = "0"
import geopandas as gpd
import typer

from dep_tools.utils import get_container_client, scale_and_offset

from azure_logger import CsvLogger, get_log_path, filter_by_log


def main(
    dataset_id: str = "mangroves", version: str = "06Sep2023", years=range(2016, 2024)
) -> None:
    aoi_by_tile = gpd.read_file(
        "https://deppcpublicstorage.blob.core.windows.net/output/aoi/aoi_split_by_landsat_pathrow.gpkg"
    ).set_index(["PATH", "ROW"], drop=False)

    prefix = f"{dataset_id}/{version}"
    logger = CsvLogger(
        name=dataset_id,
        container_client=get_container_client(),
        path=get_log_path(prefix, dataset_id, version),
        overwrite=False,
        header="time|index|status|paths|comment\n",
    )

    aoi_by_tile = filter_by_log(aoi_by_tile, logger.parse_log())
    print(list(product(aoi_by_tile.index, years)))


if __name__ == "__main__":
    typer.run(main)
