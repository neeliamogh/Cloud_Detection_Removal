from __future__ import annotations

import argparse
from urllib.parse import urlparse

import planetary_computer
import rasterio
import requests
from pystac_client import Client

from src.utils import load_config

PC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fast_check.yaml")
    parser.add_argument("--regime", default="western_ghats")
    parser.add_argument("--band", default="B04")
    args = parser.parse_args()

    cfg = load_config(args.config)
    bbox = cfg["regimes"][args.regime]["bbox"]
    client = Client.open(PC_URL)
    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f'{cfg["dates"]["start"]}/{cfg["dates"]["end"]}',
        query={"eo:cloud_cover": {"lt": 90}},
        max_items=1,
    )
    items = list(search.items())
    if not items:
        raise SystemExit("No STAC items found.")

    item = planetary_computer.sign(items[0])
    href = item.assets[args.band].href
    print("item_id:", item.id)
    print("asset_host:", urlparse(href).netloc)
    print("href_prefix:", href[:180])

    try:
        response = requests.get(href, stream=True, timeout=30)
        print("http_status:", response.status_code)
        print("content_type:", response.headers.get("content-type"))
        first = next(response.iter_content(chunk_size=128), b"")
        print("first_bytes:", first[:80])
        response.close()
    except Exception as exc:
        print("requests_error:", repr(exc))

    for candidate in [href, "/vsicurl/" + href]:
        try:
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="YES", CPL_VSIL_CURL_USE_HEAD="NO"):
                with rasterio.open(candidate) as src:
                    print("rasterio_ok:", candidate[:20], src.crs, src.width, src.height, src.count)
                    return
        except Exception as exc:
            print("rasterio_error:", candidate[:20], repr(exc))


if __name__ == "__main__":
    main()
