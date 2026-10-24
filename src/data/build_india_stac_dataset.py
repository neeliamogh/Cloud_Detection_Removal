from __future__ import annotations

import argparse
import datetime as dt
import warnings
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import numpy as np
import planetary_computer
import rasterio
import requests
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from tqdm import tqdm

from src.utils import ensure_dirs, load_config, save_json, seed_everything

PC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
READ_WARNINGS_SHOWN = 0
READ_WARNING_LIMIT = 12
FAIL_REASONS: dict[str, int] = {}


def fail(reason: str) -> None:
    FAIL_REASONS[reason] = FAIL_REASONS.get(reason, 0) + 1


def scl_cloud_mask(scl: np.ndarray, cloud_classes: list[int]) -> np.ndarray:
    mask = np.isin(scl.astype(np.uint8), np.array(cloud_classes, dtype=np.uint8))
    return mask.astype(np.float32)


def read_asset(asset_href: str, bbox: list[float], size: int, resampling=Resampling.bilinear) -> np.ndarray | None:
    global READ_WARNINGS_SHOWN
    candidates = []
    parsed = urlparse(asset_href)
    clean_href = urlunparse(parsed._replace(query=""))
    try:
        candidates.append(planetary_computer.sign(clean_href))
    except Exception as exc:
        if READ_WARNINGS_SHOWN < READ_WARNING_LIMIT:
            warnings.warn(f"Could not sign asset, trying raw href: {exc}")
            READ_WARNINGS_SHOWN += 1
    if clean_href != asset_href:
        try:
            candidates.append(planetary_computer.sign(asset_href))
        except Exception:
            pass
    candidates.append(clean_href)
    candidates.append(asset_href)
    expanded = []
    for href in candidates:
        expanded.append(href)
        if href.startswith("https://"):
            expanded.append("/vsicurl/" + href)
    candidates = list(dict.fromkeys(expanded))

    last_exc = None
    env = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "YES",
        "CPL_VSIL_CURL_USE_HEAD": "NO",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff,.TIF,.TIFF",
        "GDAL_HTTP_MAX_RETRY": "3",
        "GDAL_HTTP_RETRY_DELAY": "2",
        "AWS_NO_SIGN_REQUEST": "YES",
    }
    for href in candidates:
        try:
            with rasterio.Env(**env):
                with rasterio.open(href) as src:
                    raster_bbox = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
                    win = from_bounds(*raster_bbox, transform=src.transform)
                    arr = src.read(
                        1,
                        window=win,
                        out_shape=(size, size),
                        boundless=True,
                        fill_value=0,
                        resampling=resampling,
                    )
            if np.isfinite(arr).mean() < 0.95:
                return None
            if np.count_nonzero(arr) == 0:
                return None
            return arr.astype(np.float32)
        except Exception as exc:
            last_exc = exc
    if READ_WARNINGS_SHOWN < READ_WARNING_LIMIT:
        host = urlparse(asset_href).netloc
        warnings.warn(f"Could not read asset from {host} after signed/raw attempts: {last_exc}")
        READ_WARNINGS_SHOWN += 1
    return None


def random_patch_bbox(regime_bbox: list[float], size_deg: float, rng: np.random.Generator) -> list[float]:
    minx, miny, maxx, maxy = regime_bbox
    if (maxx - minx) <= size_deg or (maxy - miny) <= size_deg:
        return regime_bbox
    x = rng.uniform(minx, maxx - size_deg)
    y = rng.uniform(miny, maxy - size_deg)
    return [float(x), float(y), float(x + size_deg), float(y + size_deg)]


def intersect_bbox(*boxes: list[float]) -> list[float] | None:
    minx = max(box[0] for box in boxes)
    miny = max(box[1] for box in boxes)
    maxx = min(box[2] for box in boxes)
    maxy = min(box[3] for box in boxes)
    if minx >= maxx or miny >= maxy:
        return None
    return [float(minx), float(miny), float(maxx), float(maxy)]


def candidate_patch_bbox(regime_bbox: list[float], cloudy_item, clear_item, s1_item, size_deg: float, rng: np.random.Generator) -> list[float] | None:
    boxes = [regime_bbox]
    for item in [cloudy_item, clear_item, s1_item]:
        if item is not None and getattr(item, "bbox", None):
            boxes.append([float(v) for v in item.bbox])
    overlap = intersect_bbox(*boxes)
    if overlap is None:
        fail("no_scene_overlap")
        return None
    minx, miny, maxx, maxy = overlap
    if (maxx - minx) <= 0 or (maxy - miny) <= 0:
        fail("tiny_scene_overlap")
        return None
    actual_size = min(size_deg, maxx - minx, maxy - miny)
    if actual_size <= 0:
        fail("tiny_scene_overlap")
        return None
    return random_patch_bbox(overlap, actual_size, rng)


def search_items(collection: str, bbox: list[float], start: str, end: str, cloud_lt: int | None = None):
    client = Client.open(PC_URL)
    query = {}
    if cloud_lt is not None and collection == "sentinel-2-l2a":
        query["eo:cloud_cover"] = {"lt": cloud_lt}
    search = client.search(
        collections=[collection],
        bbox=bbox,
        datetime=f"{start}/{end}",
        query=query,
        max_items=500,
    )
    items = []
    for item in search.items():
        try:
            items.append(planetary_computer.sign(item))
        except Exception:
            items.append(item)
    return items


def item_date(item) -> dt.datetime:
    value = item.properties["datetime"].replace("Z", "+00:00")
    return dt.datetime.fromisoformat(value)


def read_s2_stack(item, bbox: list[float], bands: list[str], size: int) -> np.ndarray | None:
    arrays = []
    for band in bands:
        if band not in item.assets:
            return None
        arr = read_asset(item.assets[band].href, bbox, size)
        if arr is None:
            return None
        arrays.append(arr)
    return np.stack(arrays, axis=0)


def read_s1_stack(item, bbox: list[float], bands: list[str], size: int) -> np.ndarray | None:
    arrays = []
    for band in bands:
        key = band.lower()
        if key not in item.assets:
            return None
        arr = read_asset(item.assets[key].href, bbox, size)
        if arr is None:
            return None
        arrays.append(arr)
    return np.stack(arrays, axis=0)


def make_sample(regime: str, bbox: list[float], cloudy_item, clear_item, s1_item, cfg: dict, out_dir: Path, idx: int) -> dict | None:
    size = int(cfg["data"]["patch_size"])
    s2_bands = cfg["data"]["sentinel2_bands"]
    s1_bands = cfg["data"]["sentinel1_bands"]

    cloudy = read_s2_stack(cloudy_item, bbox, s2_bands, size)
    clear = read_s2_stack(clear_item, bbox, s2_bands, size)
    s1 = read_s1_stack(s1_item, bbox, s1_bands, size) if s1_item is not None else None
    if cloudy is None:
        fail("cloudy_s2_read_failed")
        return None
    if clear is None:
        fail("clear_s2_read_failed")
        return None
    if s1 is None:
        fail("s1_read_failed")
        return None

    if "SCL" not in cloudy_item.assets or "SCL" not in clear_item.assets:
        fail("missing_scl_asset")
        return None
    cloudy_scl = read_asset(cloudy_item.assets["SCL"].href, bbox, size, Resampling.nearest)
    clear_scl = read_asset(clear_item.assets["SCL"].href, bbox, size, Resampling.nearest)
    if cloudy_scl is None or clear_scl is None:
        fail("scl_read_failed")
        return None

    mask = scl_cloud_mask(cloudy_scl, cfg["data"]["cloud_scl_classes"])
    clear_mask = scl_cloud_mask(clear_scl, cfg["data"]["cloud_scl_classes"])
    cloudy_fraction = float(mask.mean())
    clear_fraction = float(clear_mask.mean())
    if not (cfg["data"]["cloudy_min_fraction"] <= cloudy_fraction <= cfg["data"]["cloudy_max_fraction"]):
        fail("cloudy_fraction_filter")
        return None
    if clear_fraction > cfg["data"]["clear_max_fraction"]:
        fail("clear_fraction_filter")
        return None

    cloudy_date = item_date(cloudy_item)
    clear_date = item_date(clear_item)
    gap = abs((cloudy_date - clear_date).days)
    if gap > int(cfg["data"]["max_temporal_gap_days"]):
        fail("temporal_gap_filter")
        return None

    path = out_dir / f"{regime}_{idx:05d}.npz"
    np.savez_compressed(
        path,
        cloudy_s2=cloudy.astype(np.float32),
        clear_s2=clear.astype(np.float32),
        s1=s1.astype(np.float32),
        cloud_mask=mask[None].astype(np.float32),
        regime=regime,
        cloudy_date=cloudy_date.isoformat(),
        clear_date=clear_date.isoformat(),
        temporal_gap_days=gap,
        cloud_fraction=cloudy_fraction,
        bbox=np.array(bbox, dtype=np.float32),
    )
    return {
        "path": str(path),
        "regime": regime,
        "cloudy_date": cloudy_date.isoformat(),
        "clear_date": clear_date.isoformat(),
        "temporal_gap_days": gap,
        "cloud_fraction": cloudy_fraction,
        "bbox": bbox,
    }


def closest_item(
    target,
    items,
    max_days: int,
    used_ids: set[str] | None = None,
    required_bbox: list[float] | None = None,
):
    if not items:
        return None
    target_date = item_date(target)
    ranked = sorted(items, key=lambda x: abs((item_date(x) - target_date).days))
    for best in ranked:
        if used_ids and best.id in used_ids:
            continue
        if required_bbox is not None:
            if not getattr(best, "bbox", None):
                continue
            if intersect_bbox(required_bbox, [float(v) for v in best.bbox]) is None:
                continue
        if abs((item_date(best) - target_date).days) <= max_days:
            return best
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    seed_everything(int(cfg["project"]["seed"]))

    out_dir = Path(cfg["paths"]["patch_dir"])
    manifest = []
    start, end = cfg["dates"]["start"], cfg["dates"]["end"]
    max_gap = int(cfg["data"]["max_temporal_gap_days"])
    target_per_regime = int(cfg["data"]["samples_per_regime"])
    min_total_samples = int(cfg["data"].get("min_total_samples", 1))
    max_scenes_per_regime = int(cfg["data"].get("max_scenes_per_regime", 160))
    patch_size_deg = float(cfg["data"].get("patch_bbox_degrees", 0.08))
    patch_attempts = int(cfg["data"].get("patch_attempts_per_scene", 3))
    rng = np.random.default_rng(int(cfg["project"]["seed"]))
    summary = []

    for regime, spec in cfg["regimes"].items():
        bbox = spec["bbox"]
        print(f"Searching regime={regime} bbox={bbox}")
        s2_cloudy = search_items("sentinel-2-l2a", bbox, start, end, cloud_lt=90)
        s2_clear = search_items("sentinel-2-l2a", bbox, start, end, cloud_lt=10)
        s1_items = search_items("sentinel-1-rtc", bbox, start, end)
        s2_cloudy = sorted(s2_cloudy, key=item_date, reverse=True)[:max_scenes_per_regime]
        s2_clear = sorted(s2_clear, key=item_date, reverse=True)
        s1_items = sorted(s1_items, key=item_date, reverse=True)
        print(f"Found S2 cloudy candidates={len(s2_cloudy)} clear={len(s2_clear)} S1={len(s1_items)}")

        count = 0
        attempted = 0
        used_clear_ids: set[str] = set()
        for cloudy in tqdm(s2_cloudy, desc=f"{regime} samples"):
            if not getattr(cloudy, "bbox", None):
                fail("cloudy_missing_bbox")
                continue
            cloudy_overlap = intersect_bbox(bbox, [float(v) for v in cloudy.bbox])
            if cloudy_overlap is None:
                fail("cloudy_no_regime_overlap")
                continue
            clear = closest_item(
                cloudy,
                s2_clear,
                max_gap,
                used_clear_ids,
                required_bbox=cloudy_overlap,
            )
            if clear is None:
                fail("no_spatial_clear_pair")
                continue
            clear_overlap = intersect_bbox(cloudy_overlap, [float(v) for v in clear.bbox])
            if clear_overlap is None:
                fail("no_clear_overlap")
                continue
            s1 = closest_item(cloudy, s1_items, 12, required_bbox=clear_overlap)
            if s1 is None:
                fail("no_spatial_s1_pair")
                continue
            if clear is None or s1 is None:
                continue
            for _ in range(patch_attempts):
                attempted += 1
                patch_bbox = candidate_patch_bbox(bbox, cloudy, clear, s1, patch_size_deg, rng)
                if patch_bbox is None:
                    continue
                sample = make_sample(regime, patch_bbox, cloudy, clear, s1, cfg, out_dir, count)
                if sample:
                    manifest.append(sample)
                    used_clear_ids.add(clear.id)
                    count += 1
                    break
            if count >= target_per_regime:
                break
        row = {
            "regime": regime,
            "saved": count,
            "attempted_patch_reads": attempted,
            "failure_reasons_so_far": dict(sorted(FAIL_REASONS.items())),
        }
        summary.append(row)
        print(f"Regime summary: {row}")

    save_json(manifest, Path(cfg["paths"]["data_root"]) / "manifest.json")
    save_json(summary, Path(cfg["paths"]["data_root"]) / "dataset_build_summary.json")
    save_json(dict(sorted(FAIL_REASONS.items())), Path(cfg["paths"]["data_root"]) / "dataset_failure_reasons.json")
    print(f"Saved {len(manifest)} real paired samples.")
    print(f"Failure reasons: {dict(sorted(FAIL_REASONS.items()))}")
    if len(manifest) == 0:
        raise RuntimeError(
            "No readable Sentinel patches were created. This is usually a remote raster access issue "
            "or overly strict filters. First run scripts/run_fast_check.sh. If it still fails, update "
            "planetary-computer/rasterio, verify network access to Azure Blob Storage, and loosen "
            "cloud thresholds in the config."
        )
    if len(manifest) < min_total_samples:
        raise RuntimeError(
            f"Only {len(manifest)} samples were created, below min_total_samples={min_total_samples}. "
            "Use fast_check first, loosen cloud filters, increase date range, or reduce holdout regimes."
        )
    if len(manifest) < 160:
        print("WARNING: Small dataset created. This run can complete, but paper-grade results need more samples.")


if __name__ == "__main__":
    main()
