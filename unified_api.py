from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import uvicorn
import io
import base64
from datetime import datetime
import numpy as np
from PIL import Image
import cv2
import uuid
import os
import sys
import asyncio
import hashlib
from pathlib import Path
import json

# Import the functions from your existing app.py
from geopy.geocoders import Nominatim
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from sentinelhub import (
    SHConfig, BBox, CRS, SentinelHubRequest, MimeType,
    DataCollection, bbox_to_dimensions, SentinelHubCatalog
)
from dotenv import load_dotenv
import pandas as pd
from data_service import data_service

# Load environment variables
load_dotenv()
CLIENT_ID = os.getenv("CLIENT_ID") or "d0769aee-284a-400d-bc5f-c9df47a197eb"
CLIENT_SECRET = os.getenv("CLIENT_SECRET") or "sKcLVzYGFc3aFaIFJTuROB1BdfAg6VZb"

# Load the TensorFlow model
model = load_model("unet_model.h5")

app = FastAPI(
    title="Unified Geospatial Change Detection API",
    description="Comprehensive satellite-based urban change detection API with both simple and advanced endpoints",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for analysis history
analysis_history = []
active_analyses = {}

# ===== PYDANTIC MODELS =====

# Simple API Models (from api_service.py)
class ChangeDetectionRequest(BaseModel):
    location: str
    zoom_level: str = "City-Wide (0.025°)"
    resolution: str = "Standard (5m)"
    alpha: float = 0.4

class ChangeDetectionResponse(BaseModel):
    success: bool
    message: str
    coordinates: Optional[Dict[str, float]] = None
    dates: Optional[Dict[str, str]] = None
    statistics: Optional[Dict[str, Any]] = None
    images: Optional[Dict[str, str]] = None
    socioeconomic_data: Optional[Dict[str, Any]] = None  # Added for datasets
    real_estate_data: Optional[Dict[str, Any]] = None    # Added for datasets
    comprehensive_analysis: Optional[Dict[str, Any]] = None  # Added for analysis

# Advanced API Models (from geospatial agent)
class AnalysisRequest(BaseModel):
    location: Dict[str, float]  # {"lat": 25.2048, "lon": 55.2708}
    time_range: Optional[Dict[str, str]] = None
    analysis_type: str = "urban_change"
    zoom_level: Optional[str] = "City-Wide (0.025°)"
    resolution: Optional[str] = "Standard (5m)"
    overlay_alpha: Optional[float] = 0.4
    include_images: Optional[bool] = True

class BatchAnalysisRequest(BaseModel):
    locations: List[Dict[str, Any]]
    zoom_level: Optional[str] = "City-Wide (0.025°)"
    resolution: Optional[str] = "Standard (5m)"
    overlay_alpha: Optional[float] = 0.4
    include_images: Optional[bool] = True

class LocationSearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 10

class AnalysisResponse(BaseModel):
    status: str
    data: Dict[str, Any]

class BatchAnalysisResponse(BaseModel):
    status: str
    batch_id: str
    total_locations: int
    completed: int
    results: List[Dict[str, Any]]

class LocationSearchResponse(BaseModel):
    status: str
    locations: List[Dict[str, Any]]

class AnalysisHistoryResponse(BaseModel):
    status: str
    analyses: List[Dict[str, Any]]

# NDVI Analysis Models
class NDVIAnalysisRequest(BaseModel):
    location: str = Field(..., min_length=1, max_length=200)
    timeline_start: Optional[str] = None
    timeline_end: Optional[str] = None
    zoom_level: str = Field(default="City-Wide (0.025°)")
    resolution: str = Field(default="Standard (5m)")
    want_recommendations: Optional[bool] = False
    want_visualizations: Optional[bool] = True
    analysis_focus: str = Field(default="vegetation", description="Focus area: vegetation, urban, water, or general")

class NDVIAnalysisResponse(BaseModel):
    success: bool
    location: str
    coordinates: Dict[str, float]
    timeline_start: str
    timeline_end: str
    chosen_dates: List[Dict[str, str]]
    available_dates_count: int
    timestamp: str
    ndvi_analysis: Dict[str, Any]
    change_analysis: Dict[str, Any]
    recommendations: List[str] = []
    visualizations: Dict[str, str] = {}
    socioeconomic_correlation: Optional[Dict[str, Any]] = None

# ===== UTILITY FUNCTIONS =====

def get_coordinates(location_name):
    geolocator = Nominatim(user_agent="unified_change_detector_app")
    location = geolocator.geocode(location_name)
    if location:
        return location.latitude, location.longitude
    return None, None

def map_bbox_choice(choice):
    return {"City-Wide (0.025°)": 0.025, "Block-Level (0.01°)": 0.01, "Zoomed-In (0.005°)": 0.005}[choice]

def map_resolution_choice(choice):
    return {"Coarse (10m)": 10, "Standard (5m)": 5, "Fine (2.5m)": 2.5}[choice]

def get_two_low_cloud_dates(lat, lon, buffer):
    config = SHConfig()
    config.sh_client_id = CLIENT_ID
    config.sh_client_secret = CLIENT_SECRET

    bbox = BBox([lon - buffer, lat - buffer, lon + buffer, lat + buffer], crs=CRS.WGS84)
    catalog = SentinelHubCatalog(config=config)

    start_date = datetime(2017, 3, 28)
    end_date = datetime.now()

    search_iterator = catalog.search(
        DataCollection.SENTINEL2_L2A,
        bbox=bbox,
        time=(start_date, end_date),
        filter={
            "op": "<=",
            "args": [
                {"property": "eo:cloud_cover"},
                45.0
            ]
        },
        filter_lang="cql2-json",
        fields={
            "include": ["id", "properties.datetime", "properties.eo:cloud_cover"],
            "exclude": []
        }
    )

    results = list(search_iterator)
    if len(results) < 2:
        return None, None

    sorted_results = sorted(results, key=lambda r: r["properties"]["datetime"])
    date1 = sorted_results[0]["properties"]["datetime"][:10]
    date2 = sorted_results[-1]["properties"]["datetime"][:10]
    return date1, date2

def normalize_image(data):
    p2 = np.percentile(data, 2)
    p98 = np.percentile(data, 98)
    scaled = np.clip((data - p2) / (p98 - p2), 0, 1)
    return (scaled * 255).astype(np.uint8)

def fetch_sentinel_image(lat, lon, date, buffer, resolution):
    config = SHConfig()
    config.sh_client_id = CLIENT_ID
    config.sh_client_secret = CLIENT_SECRET

    bbox = BBox([lon - buffer, lat - buffer, lon + buffer, lat + buffer], crs=CRS.WGS84)
    size = bbox_to_dimensions(bbox, resolution=resolution)

    request = SentinelHubRequest(
        evalscript="""
        //VERSION=3
        function setup() {
          return {
            input: ["B04", "B03", "B02"],
            output: { bands: 3, sampleType: "AUTO" }
          };
        }
        function evaluatePixel(sample) {
          return [sample.B04, sample.B03, sample.B02];
        }
        """,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(date, date)
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config
    )

    try:
        data = request.get_data()[0]
        image = normalize_image(data)
        return Image.fromarray(image)
    except Exception as e:
        print(f"[ERROR] Image fetch failed: {e}")
        return None

def predict_change_mask(before_img, after_img):
    before_resized = before_img.resize((128, 128))
    after_resized = after_img.resize((128, 128))

    before_array = np.array(before_resized) / 255.0
    after_array = np.array(after_resized) / 255.0

    combined = np.concatenate([before_array, after_array], axis=-1)
    input_tensor = np.expand_dims(combined, axis=0)

    prediction = model.predict(input_tensor)[0, :, :, 0]
    binary_mask = (prediction > 0.5).astype(np.uint8) * 255
    return Image.fromarray(binary_mask).resize(before_img.size)

def overlay_mask_on_image(base_img, mask_img, color=(255, 0, 0), alpha=0.4):
    base = np.array(base_img.convert("RGB")).astype(np.uint8)
    mask = np.array(mask_img.resize(base_img.size)).astype(np.uint8)

    overlay = np.zeros_like(base)
    overlay[mask > 0] = color

    blended = (base * (1 - alpha) + overlay * alpha).astype(np.uint8)

    # Add contours for clear boundaries
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (255, 255, 0), thickness=2)  # yellow borders

    return Image.fromarray(blended)

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode()

def image_to_base64_with_prefix(image: Image.Image) -> str:
    """Convert PIL Image to base64 string with data URL prefix"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two coordinates in kilometers."""
    from math import radians, cos, sin, asin, sqrt

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

def get_available_dates(lat: float, lon: float, buffer: float = 0.025) -> List[str]:
    """Get available satellite imagery dates for a location."""
    try:
        config = SHConfig()
        config.sh_client_id = CLIENT_ID
        config.sh_client_secret = CLIENT_SECRET

        bbox = BBox([lon - buffer, lat - buffer, lon + buffer, lat + buffer], crs=CRS.WGS84)
        catalog = SentinelHubCatalog(config=config)

        start_date = datetime(2017, 3, 28)
        end_date = datetime.now()

        search_iterator = catalog.search(
            DataCollection.SENTINEL2_L2A,
            bbox=bbox,
            time=(start_date, end_date),
            filter={
                "op": "<=",
                "args": [
                    {"property": "eo:cloud_cover"},
                    45.0
                ]
            },
            filter_lang="cql2-json",
            fields={
                "include": ["id", "properties.datetime", "properties.eo:cloud_cover"],
                "exclude": []
            }
        )

        results = list(search_iterator)
        dates = [r["properties"]["datetime"][:10] for r in results]
        return sorted(list(set(dates)))
    except Exception as e:
        print(f"Error getting available dates: {e}")
        return []

def search_locations(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search for locations and return detailed information."""
    try:
        geolocator = Nominatim(user_agent="unified_change_detector_app")
        locations = []
        geocode_results = geolocator.geocode(query, exactly_one=False, limit=limit)

        if geocode_results:
            for result in geocode_results[:limit]:
                location_info = {
                    "name": result.address,
                    "display_name": result.address,
                    "coordinates": {
                        "lat": result.latitude,
                        "lon": result.longitude
                    },
                    "bbox": result.raw.get("boundingbox", None) if hasattr(result, 'raw') else None,
                    "place_type": result.raw.get("type", "unknown") if hasattr(result, 'raw') else "unknown",
                    "country": result.raw.get("display_name", "").split(", ")[-1] if hasattr(result, 'raw') else "unknown"
                }
                locations.append(location_info)

        return locations
    except Exception as e:
        print(f"Location search error: {e}")
        return []

# NDVI Analysis Utility Functions
CACHE_DIR = Path("sentinel_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Thresholds for change detection
DNDVI_T = 0.07
DNDBI_T = 0.07
DNDWI_T = 0.07
VALID_MIN = 0.01

def _sentinel_cache_key(lat: float, lon: float, date: str, buf: float, res: float) -> Path:
    h = hashlib.sha256(f"{lat:.6f},{lon:.6f},{date},{buf},{res}".encode()).hexdigest()
    return CACHE_DIR / f"{h}.npz"

def _normalize_for_ndvi(img: np.ndarray) -> np.ndarray:
    p2, p98 = np.percentile(img, (2, 98))
    return np.clip((img - p2) / (p98 - p2 + 1e-9), 0, 1)

def find_smart_dates(lat: float, lon: float, buf: float,
                     t0: Optional[str], t1: Optional[str]) -> tuple[str, str, List[Dict[str, str]]]:
    """Find optimal dates for NDVI analysis with low cloud cover and good temporal separation"""
    config = SHConfig()
    config.sh_client_id = CLIENT_ID
    config.sh_client_secret = CLIENT_SECRET

    cat = SentinelHubCatalog(config=config)
    bbox = BBox([lon - buf, lat - buf, lon + buf, lat + buf], crs=CRS.WGS84)
    time_from = datetime.strptime(t0, "%Y-%m-%d") if t0 else datetime(2018, 1, 1)
    time_to = datetime.strptime(t1, "%Y-%m-%d") if t1 else datetime.now()

    results = cat.search(
        DataCollection.SENTINEL2_L2A,
        bbox=bbox,
        time=(time_from, time_to),
        fields={"include": ["properties.datetime", "properties.eo:cloud_cover"], "exclude": []},
    )

    # Collect dates with cloud cover
    rows = []
    for r in results:
        dt = r["properties"]["datetime"][:10]
        cc = r["properties"].get("eo:cloud_cover")
        try:
            cc_val = float(cc) if cc is not None else 100.0
        except Exception:
            cc_val = 100.0
        rows.append((dt, cc_val))

    if not rows:
        return None, None, []

    # Group by unique date, keep min cloud cover per date
    by_date: Dict[str, float] = {}
    for dt, cc in rows:
        if (dt not in by_date) or (cc < by_date[dt]):
            by_date[dt] = cc

    # Sort by date and create list with cloud cover
    sorted_dates = sorted(by_date.items(), key=lambda x: x[0])
    all_dates = [{"date": d, "cloud": f"{c:.1f}"} for d, c in sorted_dates]

    # Split into quartiles, pick least-cloudy in first and last quartiles
    n = len(sorted_dates)
    q = max(1, n // 4)
    first_q = sorted_dates[:q] if n >= 4 else sorted_dates[: max(1, n // 2)]
    last_q = sorted_dates[-q:] if n >= 4 else sorted_dates[-max(1, n // 2):]

    before = min(first_q, key=lambda x: x[1])  # (date, cloud)
    after = min(last_q, key=lambda x: x[1])

    return before[0], after[0], all_dates

async def fetch_ndvi_image(lat: float, lon: float, date: str, buf: float, res_m: float) -> Optional[dict]:
    """Fetch satellite image with bands needed for NDVI analysis"""
    cache_path = _sentinel_cache_key(lat, lon, date, buf, res_m)
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=True) as npz:
                raw = npz["raw"]
                rgb = Image.fromarray(npz["rgb"])
            return {"raw": raw, "rgb": rgb}
        except Exception as e:
            print(f"Cache load failed for {date}: {e}")
            try:
                cache_path.unlink()
            except Exception:
                pass

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _blocking_fetch_ndvi, lat, lon, date, buf, res_m)
    if result:
        try:
            np.savez_compressed(cache_path, raw=result["raw"], rgb=np.array(result["rgb"]))
        except Exception as e:
            print(f"Cache save failed for {date}: {e}")
    return result

def _blocking_fetch_ndvi(lat: float, lon: float, date: str, buf: float, res_m: float) -> Optional[dict]:
    """Fetch satellite data with bands for NDVI calculation"""
    config = SHConfig()
    config.sh_client_id = CLIENT_ID
    config.sh_client_secret = CLIENT_SECRET

    bbox = BBox([lon - buf, lat - buf, lon + buf, lat + buf], crs=CRS.WGS84)
    dims = bbox_to_dimensions(bbox, resolution=res_m)
    dims = (max(128, min(dims[0], 2048)), max(128, min(dims[1], 2048)))

    evalscript = """
    //VERSION=3
    function setup() {
      return {
        input: [{ bands: ["B02","B03","B04","B08","B11"], units: "DN" }],
        output: { bands: 5, sampleType: "FLOAT32" }
      };
    }
    function evaluatePixel(s) {
      return [s.B02/10000, s.B03/10000, s.B04/10000, s.B08/10000, s.B11/10000];
    }
    """

    req = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(f"{date}T00:00:00Z", f"{date}T23:59:59Z"),
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=dims,
        config=config,
    )

    try:
        data = req.get_data()
        if not data:
            return None

        raw = data[0]
        rgb = (_normalize_for_ndvi(raw[:, :, [2, 1, 0]]) * 255).astype(np.uint8)
        return {"raw": raw, "rgb": Image.fromarray(rgb)}
    except Exception as e:
        print(f"NDVI image fetch failed: {e}")
        return None

def to_f32(a: np.ndarray) -> np.ndarray:
    return np.clip(a.astype(np.float32), 1e-6, None)

def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    return (nir - red) / (nir + red + 1e-6)

def ndbi(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return (swir - nir) / (swir + nir + 1e-6)

def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return (green - nir) / (green + nir + 1e-6)

def compute_ndvi_analysis(b: np.ndarray, a: np.ndarray) -> Dict[str, Any]:
    """Compute comprehensive NDVI analysis including change detection"""
    (B02b, B03b, B04b, B08b, B11b) = [to_f32(b[:, :, i]) for i in range(5)]
    (B02a, B03a, B04a, B08a, B11a) = [to_f32(a[:, :, i]) for i in range(5)]

    # Calculate indices for before and after
    ndvi_before = ndvi(B08b, B04b)
    ndvi_after = ndvi(B08a, B04a)
    ndbi_before = ndbi(B11b, B08b)
    ndbi_after = ndbi(B11a, B08a)
    ndwi_before = ndwi(B03b, B08b)
    ndwi_after = ndwi(B03a, B08a)

    # Calculate changes
    d_ndvi = ndvi_after - ndvi_before
    d_ndbi = ndbi_after - ndbi_before
    d_ndwi = ndwi_after - ndwi_before

    valid = ((B02a + B03a + B04a + B08a + B11a) / 5) > VALID_MIN

    # Change masks
    veg_gain = (d_ndvi >= DNDVI_T) & valid
    veg_loss = (d_ndvi <= -DNDVI_T) & valid
    urb_gain = (d_ndbi >= DNDBI_T) & (d_ndvi <= 0) & valid
    urb_loss = (d_ndbi <= -DNDBI_T) & valid
    wat_gain = (d_ndwi >= DNDWI_T) & valid
    wat_loss = (d_ndwi <= -DNDWI_T) & valid

    # Statistics
    total_pixels = int(valid.sum())
    veg_gain_count = int(veg_gain.sum())
    veg_loss_count = int(veg_loss.sum())
    urb_gain_count = int(urb_gain.sum())
    urb_loss_count = int(urb_loss.sum())
    wat_gain_count = int(wat_gain.sum())
    wat_loss_count = int(wat_loss.sum())

    # NDVI statistics
    ndvi_before_stats = {
        "mean": float(np.mean(ndvi_before[valid])),
        "std": float(np.std(ndvi_before[valid])),
        "min": float(np.min(ndvi_before[valid])),
        "max": float(np.max(ndvi_before[valid])),
        "median": float(np.median(ndvi_before[valid]))
    }

    ndvi_after_stats = {
        "mean": float(np.mean(ndvi_after[valid])),
        "std": float(np.std(ndvi_after[valid])),
        "min": float(np.min(ndvi_after[valid])),
        "max": float(np.max(ndvi_after[valid])),
        "median": float(np.median(ndvi_after[valid]))
    }

    return {
        "ndvi_before": ndvi_before,
        "ndvi_after": ndvi_after,
        "ndvi_change": d_ndvi,
        "change_masks": {
            "vegetation_gain": veg_gain,
            "vegetation_loss": veg_loss,
            "urbanization": urb_gain,
            "urban_loss": urb_loss,
            "water_gain": wat_gain,
            "water_loss": wat_loss
        },
        "statistics": {
            "total_valid_pixels": total_pixels,
            "vegetation_gain": {"count": veg_gain_count, "percentage": veg_gain_count/total_pixels*100 if total_pixels > 0 else 0},
            "vegetation_loss": {"count": veg_loss_count, "percentage": veg_loss_count/total_pixels*100 if total_pixels > 0 else 0},
            "urbanization": {"count": urb_gain_count, "percentage": urb_gain_count/total_pixels*100 if total_pixels > 0 else 0},
            "urban_loss": {"count": urb_loss_count, "percentage": urb_loss_count/total_pixels*100 if total_pixels > 0 else 0},
            "water_gain": {"count": wat_gain_count, "percentage": wat_gain_count/total_pixels*100 if total_pixels > 0 else 0},
            "water_loss": {"count": wat_loss_count, "percentage": wat_loss_count/total_pixels*100 if total_pixels > 0 else 0}
        },
        "ndvi_statistics": {
            "before": ndvi_before_stats,
            "after": ndvi_after_stats,
            "change": {
                "mean_change": float(np.mean(d_ndvi[valid])),
                "std_change": float(np.std(d_ndvi[valid])),
                "significant_change_pixels": int(np.sum(np.abs(d_ndvi[valid]) >= DNDVI_T))
            }
        }
    }

def create_ndvi_visualizations(analysis_data: Dict[str, Any], after_img: Image.Image) -> Dict[str, str]:
    """Create NDVI visualization overlays"""
    visualizations = {}

    # NDVI before/after heatmaps
    ndvi_before = analysis_data["ndvi_before"]
    ndvi_after = analysis_data["ndvi_after"]
    ndvi_change = analysis_data["ndvi_change"]

    # NDVI before visualization (green scale)
    ndvi_before_norm = np.clip((ndvi_before + 1) / 2, 0, 1)  # Normalize -1 to 1 -> 0 to 1
    ndvi_before_rgb = np.zeros((*ndvi_before.shape, 3), dtype=np.uint8)
    ndvi_before_rgb[:, :, 1] = (ndvi_before_norm * 255).astype(np.uint8)  # Green channel
    visualizations["ndvi_before"] = image_to_base64(Image.fromarray(ndvi_before_rgb))

    # NDVI after visualization (green scale)
    ndvi_after_norm = np.clip((ndvi_after + 1) / 2, 0, 1)
    ndvi_after_rgb = np.zeros((*ndvi_after.shape, 3), dtype=np.uint8)
    ndvi_after_rgb[:, :, 1] = (ndvi_after_norm * 255).astype(np.uint8)
    visualizations["ndvi_after"] = image_to_base64(Image.fromarray(ndvi_after_rgb))

    # NDVI change visualization (red-green scale)
    ndvi_change_norm = np.clip((ndvi_change + 1) / 2, 0, 1)
    ndvi_change_rgb = np.zeros((*ndvi_change.shape, 3), dtype=np.uint8)
    ndvi_change_rgb[:, :, 0] = ((1 - ndvi_change_norm) * 255).astype(np.uint8)  # Red for loss
    ndvi_change_rgb[:, :, 1] = (ndvi_change_norm * 255).astype(np.uint8)       # Green for gain
    visualizations["ndvi_change"] = image_to_base64(Image.fromarray(ndvi_change_rgb))

    # Change overlay on satellite image
    change_masks = analysis_data["change_masks"]
    base = np.array(after_img.resize((ndvi_change.shape[1], ndvi_change.shape[0])).convert("RGB"))
    overlay = np.zeros_like(base)

    # Color coding for different changes
    overlay[change_masks["vegetation_gain"]] = [0, 200, 0]      # Bright green
    overlay[change_masks["vegetation_loss"]] = [200, 0, 0]      # Red
    overlay[change_masks["urbanization"]] = [128, 128, 128]     # Gray
    overlay[change_masks["water_gain"]] = [0, 0, 200]          # Blue
    overlay[change_masks["water_loss"]] = [200, 200, 0]        # Yellow

    alpha = 0.45
    blended = (base * (1 - alpha) + overlay * alpha).astype(np.uint8)
    visualizations["change_overlay"] = image_to_base64(Image.fromarray(blended))

    return visualizations

def parse_intent(query: str) -> Dict[str, bool]:
    """Parse user intent from query"""
    q_low = query.lower()
    intents = {
        "vegetation_focus": any(k in q_low for k in ["deforest", "forest", "tree", "green cover", "ndvi", "vegetation"]),
        "urban_focus": any(k in q_low for k in ["urban", "built-up", "construction", "sprawl", "ndbi"]),
        "water_focus": any(k in q_low for k in ["flood", "water", "wetland", "reservoir", "lake", "ndwi"]),
    }
    if not any(intents.values()):
        intents["general"] = True
    else:
        intents["general"] = False
    return intents

def generate_ndvi_recommendations(analysis_data: Dict[str, Any], intents: Dict[str, bool], location: str) -> List[str]:
    """Generate recommendations based on NDVI analysis"""
    recommendations = []
    stats = analysis_data["statistics"]

    if intents.get("vegetation_focus"):
        veg_loss_pct = stats["vegetation_loss"]["percentage"]
        veg_gain_pct = stats["vegetation_gain"]["percentage"]

        if veg_loss_pct > 5:
            recommendations.append(f"Significant vegetation loss detected ({veg_loss_pct:.1f}%). Consider implementing reforestation programs.")
        if veg_gain_pct > 3:
            recommendations.append(f"Positive vegetation growth observed ({veg_gain_pct:.1f}%). Monitor and protect these recovering areas.")
        if veg_loss_pct > veg_gain_pct * 2:
            recommendations.append("Vegetation loss exceeds growth. Urgent conservation measures may be needed.")

    if intents.get("urban_focus"):
        urban_growth_pct = stats["urbanization"]["percentage"]
        if urban_growth_pct > 3:
            recommendations.append(f"Rapid urbanization detected ({urban_growth_pct:.1f}%). Ensure sustainable development practices.")
        if urban_growth_pct > 1:
            recommendations.append("Monitor infrastructure development to balance growth with environmental protection.")

    if intents.get("water_focus"):
        water_loss_pct = stats["water_loss"]["percentage"]
        water_gain_pct = stats["water_gain"]["percentage"]

        if water_loss_pct > 2:
            recommendations.append(f"Water body reduction detected ({water_loss_pct:.1f}%). Investigate potential drought or diversion issues.")
        if water_gain_pct > 2:
            recommendations.append(f"New water bodies detected ({water_gain_pct:.1f}%). Monitor for flooding or new water management projects.")

    # General recommendations
    total_change = sum(s["percentage"] for s in stats.values() if isinstance(s, dict) and "percentage" in s)
    if total_change > 10:
        recommendations.append("Significant landscape changes detected. Consider comprehensive environmental impact assessment.")

    if len(recommendations) == 0:
        recommendations.append("Landscape appears relatively stable. Continue regular monitoring for early change detection.")

    return recommendations[:5]  # Limit to 5 recommendations

# ===== API ENDPOINTS =====

@app.get("/")
async def root():
    return {
        "message": "Unified Geospatial Change Detection API",
        "status": "running",
        "version": "2.0.0",
        "endpoints": {
            "simple_api": {
                "detect_change": "/detect-change",
                "get_image": "/detect-change/{location}/images/{image_type}",
                "get_coordinates": "/locations/{location}/coordinates"
            },
            "advanced_api": {
                "analyze": "/analyze",
                "analyze_by_location": "/analyze/location",
                "batch_analyze": "/analyze/batch",
                "search_locations": "/locations/search",
                "history": "/analyze/history"
            },
            "ndvi_analysis": {
                "comprehensive_ndvi": "/analyze/ndvi",
                "quick_ndvi": "/analyze/ndvi/{location}/quick",
                "description": "NDVI-based vegetation change detection with socioeconomic correlation"
            },
            "data_endpoints": {
                "socioeconomic": "/locations/{location}/socioeconomic",
                "zip_analysis": "/zip-codes/{zip_code}/analysis",
                "dataset_info": "/datasets/info"
            }
        },
        "features": {
            "ndvi_analysis": [
                "Vegetation gain/loss detection",
                "Urban expansion tracking",
                "Water body changes",
                "Smart date selection",
                "NDVI visualizations",
                "Socioeconomic correlation",
                "Actionable recommendations"
            ]
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "sentinel_hub_configured": CLIENT_ID is not None and CLIENT_SECRET is not None
    }

# ===== SIMPLE API ENDPOINTS (Original from api_service.py) =====

@app.post("/detect-change", response_model=ChangeDetectionResponse)
async def detect_change(request: ChangeDetectionRequest):
    """
    Simple change detection analysis for a given location name
    """
    try:
        # Get coordinates from location name
        lat, lon = get_coordinates(request.location)
        if lat is None:
            return ChangeDetectionResponse(
                success=False,
                message="Location not found"
            )

        # Map parameters
        buffer = map_bbox_choice(request.zoom_level)
        resolution = map_resolution_choice(request.resolution)

        # Get satellite imagery dates
        date1, date2 = get_two_low_cloud_dates(lat, lon, buffer)
        if not date1 or not date2:
            return ChangeDetectionResponse(
                success=False,
                message="Not enough 0% cloud images found for this location"
            )

        # Fetch satellite images
        before_img = fetch_sentinel_image(lat, lon, date1, buffer, resolution)
        after_img = fetch_sentinel_image(lat, lon, date2, buffer, resolution)

        if not before_img or not after_img:
            return ChangeDetectionResponse(
                success=False,
                message="Failed to fetch satellite images"
            )

        # Perform change detection
        mask = predict_change_mask(before_img, after_img)
        overlayed = overlay_mask_on_image(after_img, mask, alpha=request.alpha)

        # Calculate statistics
        mask_array = np.array(mask)
        changed_pixels = int(np.sum(mask_array > 0))
        total_pixels = mask_array.size
        change_percentage = (changed_pixels / total_pixels) * 100

        # Convert images to base64
        images = {
            "before": image_to_base64(before_img),
            "after": image_to_base64(after_img),
            "mask": image_to_base64(mask),
            "overlay": image_to_base64(overlayed)
        }

        # Get comprehensive socioeconomic and real estate analysis
        comprehensive_data = data_service.get_comprehensive_analysis(lat, lon, request.location)

        return ChangeDetectionResponse(
            success=True,
            message="Change detection completed successfully",
            coordinates={"latitude": lat, "longitude": lon},
            dates={"before": date1, "after": date2},
            statistics={
                "changed_pixels": changed_pixels,
                "total_pixels": total_pixels,
                "change_percentage": round(change_percentage, 2)
            },
            images=images,
            socioeconomic_data=comprehensive_data.get("census_data"),
            real_estate_data=comprehensive_data.get("real_estate_data"),
            comprehensive_analysis=comprehensive_data.get("analysis")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/detect-change/{location}/images/{image_type}")
async def get_change_image(location: str, image_type: str, zoom_level: str = "City-Wide (0.025°)",
                          resolution: str = "Standard (5m)", alpha: float = 0.4):
    """
    Get a specific image from change detection analysis as a PNG response
    """
    if image_type not in ['before', 'after', 'mask', 'overlay']:
        raise HTTPException(status_code=400, detail="Invalid image type")

    try:
        lat, lon = get_coordinates(location)
        if lat is None:
            raise HTTPException(status_code=404, detail="Location not found")

        buffer = map_bbox_choice(zoom_level)
        resolution_val = map_resolution_choice(resolution)

        date1, date2 = get_two_low_cloud_dates(lat, lon, buffer)
        if not date1 or not date2:
            raise HTTPException(status_code=404, detail="No suitable satellite images found")

        before_img = fetch_sentinel_image(lat, lon, date1, buffer, resolution_val)
        after_img = fetch_sentinel_image(lat, lon, date2, buffer, resolution_val)

        if not before_img or not after_img:
            raise HTTPException(status_code=500, detail="Failed to fetch satellite images")

        # Get the requested image
        if image_type == "before":
            image = before_img
        elif image_type == "after":
            image = after_img
        elif image_type == "mask":
            image = predict_change_mask(before_img, after_img)
        elif image_type == "overlay":
            mask = predict_change_mask(before_img, after_img)
            image = overlay_mask_on_image(after_img, mask, alpha=alpha)

        # Convert to bytes and return as streaming response
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_buffer.seek(0)

        return StreamingResponse(
            io.BytesIO(img_buffer.getvalue()),
            media_type="image/png",
            headers={"Content-Disposition": f"inline; filename={location}_{image_type}.png"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

@app.get("/locations/{location}/coordinates")
async def get_location_coordinates(location: str):
    """Get coordinates for a location name"""
    lat, lon = get_coordinates(location)
    if lat is None:
        raise HTTPException(status_code=404, detail="Location not found")

    return {
        "location": location,
        "latitude": lat,
        "longitude": lon
    }

# ===== ADVANCED API ENDPOINTS (From geospatial agent) =====

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    """
    Advanced geospatial analysis with detailed response format
    """
    try:
        # Extract coordinates
        if "lat" not in request.location or "lon" not in request.location:
            raise HTTPException(status_code=400, detail="Location must contain 'lat' and 'lon' fields")

        lat = request.location["lat"]
        lon = request.location["lon"]

        # Map parameters
        buffer = map_bbox_choice(request.zoom_level)
        resolution = map_resolution_choice(request.resolution)

        # Get satellite imagery dates
        date1, date2 = get_two_low_cloud_dates(lat, lon, buffer)
        if not date1 or not date2:
            raise HTTPException(status_code=404, detail="Not enough suitable satellite images found")

        # Fetch satellite images
        before_img = fetch_sentinel_image(lat, lon, date1, buffer, resolution)
        after_img = fetch_sentinel_image(lat, lon, date2, buffer, resolution)

        if not before_img or not after_img:
            raise HTTPException(status_code=500, detail="Failed to fetch satellite images")

        # Perform change detection
        mask = predict_change_mask(before_img, after_img)
        overlayed = overlay_mask_on_image(after_img, mask, alpha=request.overlay_alpha)

        # Calculate statistics
        mask_array = np.array(mask)
        changed_pixels = int(np.sum(mask_array > 0))
        total_pixels = mask_array.size
        change_percentage = (changed_pixels / total_pixels) * 100

        # Generate job ID
        job_id = str(uuid.uuid4())

        # Convert to the advanced API format
        response_data = {
            "jobId": job_id,
            "status": "COMPLETE",
            "progress": "Analysis complete",
            "elapsedTime": 30000,
            "data": {
                "type": "change_detection_analysis",
                "intent": {
                    "intent": "CHANGE_DETECTION",
                    "location": f"Lat: {lat}, Lon: {lon}",
                    "dateRange": [date1, date2],
                    "confidence": 95,
                    "extractedParams": {
                        "priceThreshold": None,
                        "riskCriteria": True,
                        "timeFrame": None
                    }
                },
                "data": {
                    "changePolygons": {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": [[
                                        [lon - buffer, lat - buffer],
                                        [lon + buffer, lat - buffer],
                                        [lon + buffer, lat + buffer],
                                        [lon - buffer, lat + buffer],
                                        [lon - buffer, lat - buffer]
                                    ]]
                                },
                                "properties": {
                                    "changeType": "land_use_change",
                                    "confidence": 0.85,
                                    "area": changed_pixels,
                                    "detectionMethod": "satellite_analysis"
                                }
                            }
                        ]
                    },
                    "statistics": {
                        "totalChangeArea": changed_pixels,
                        "changePercentage": change_percentage,
                        "changedPixels": changed_pixels,
                        "totalPixels": total_pixels
                    },
                    "images": {
                        "beforeImage": image_to_base64_with_prefix(before_img),
                        "afterImage": image_to_base64_with_prefix(after_img),
                        "overlayImage": image_to_base64_with_prefix(overlayed),
                        "maskImage": image_to_base64_with_prefix(mask)
                    },
                    "metadata": {
                        "location": f"Lat: {lat}, Lon: {lon}",
                        "dateRange": [date1, date2],
                        "resolution": request.resolution or "Standard (5m)",
                        "algorithm": "U-Net Change Detection",
                        "analysisDate": datetime.now().isoformat()
                    }
                },
                "placeholder": False
            }
        }

        # Store analysis in history
        analysis_record = {
            "id": job_id,
            "timestamp": datetime.now().isoformat(),
            "location": request.location,
            "parameters": {
                "zoom_level": request.zoom_level,
                "resolution": request.resolution,
                "overlay_alpha": request.overlay_alpha
            },
            "result": {
                "coordinates": {"lat": lat, "lon": lon},
                "dates": {"before": date1, "after": date2},
                "statistics": {
                    "changed_pixels": changed_pixels,
                    "total_pixels": total_pixels,
                    "change_percentage": round(change_percentage, 2)
                },
                "images": {
                    "before": image_to_base64(before_img),
                    "after": image_to_base64(after_img),
                    "overlay": image_to_base64(overlayed),
                    "mask": image_to_base64(mask)
                }
            }
        }
        analysis_history.append(analysis_record)

        return response_data

    except Exception as e:
        job_id = str(uuid.uuid4())
        return {
            "jobId": job_id,
            "status": "ERROR",
            "progress": f"Analysis failed: {str(e)}",
            "elapsedTime": 0,
            "data": {
                "type": "change_detection_analysis",
                "intent": {
                    "intent": "CHANGE_DETECTION",
                    "location": "Unknown",
                    "dateRange": ["", ""],
                    "confidence": 0,
                    "extractedParams": {
                        "priceThreshold": None,
                        "riskCriteria": False,
                        "timeFrame": None
                    }
                },
                "data": {
                    "changePolygons": {"type": "FeatureCollection", "features": []},
                    "statistics": {"totalChangeArea": 0, "changePercentage": 0, "changedPixels": 0, "totalPixels": 0},
                    "images": {"beforeImage": "", "afterImage": "", "overlayImage": "", "maskImage": ""},
                    "metadata": {
                        "location": "Unknown",
                        "dateRange": ["", ""],
                        "resolution": "Unknown",
                        "algorithm": "U-Net Change Detection",
                        "analysisDate": datetime.now().isoformat(),
                        "error": str(e)
                    }
                },
                "placeholder": True
            }
        }

@app.post("/analyze/location")
async def analyze_by_location_name(request: dict):
    """
    Perform geospatial analysis using a location name instead of coordinates
    """
    try:
        location_name = request.get("location_name")
        if not location_name:
            raise HTTPException(status_code=400, detail="location_name is required")

        lat, lon = get_coordinates(location_name)
        if lat is None or lon is None:
            raise HTTPException(status_code=404, detail="Location not found")

        analysis_request = AnalysisRequest(
            location={"lat": lat, "lon": lon},
            zoom_level=request.get("zoom_level", "City-Wide (0.025°)"),
            resolution=request.get("resolution", "Standard (5m)"),
            overlay_alpha=request.get("overlay_alpha", 0.4)
        )

        return await analyze(analysis_request)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/locations/search", response_model=LocationSearchResponse)
async def search_locations_endpoint(request: LocationSearchRequest):
    """
    Search for locations by name or address
    """
    try:
        locations = search_locations(request.query, request.limit)
        return LocationSearchResponse(
            status="success",
            locations=locations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Location search failed: {str(e)}")

@app.get("/analyze/history", response_model=AnalysisHistoryResponse)
async def get_analysis_history(limit: int = 50, offset: int = 0):
    """
    Get analysis history with pagination
    """
    try:
        sorted_history = sorted(analysis_history, key=lambda x: x["timestamp"], reverse=True)
        paginated_history = sorted_history[offset:offset + limit]

        return AnalysisHistoryResponse(
            status="success",
            analyses=paginated_history
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")

@app.get("/analyze/history/{analysis_id}")
async def get_analysis_by_id(analysis_id: str):
    """
    Get a specific analysis by ID
    """
    analysis = next((a for a in analysis_history if a["id"] == analysis_id), None)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return AnalysisResponse(
        status="success",
        data=analysis
    )

@app.delete("/analyze/history/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """
    Delete a specific analysis from history
    """
    global analysis_history
    analysis_history = [a for a in analysis_history if a["id"] != analysis_id]
    return {"status": "success", "message": "Analysis deleted"}

@app.get("/system/info")
async def get_system_info():
    """
    Get system information and available options
    """
    return {
        "status": "success",
        "system_info": {
            "api_version": "2.0.0",
            "model_loaded": model is not None,
            "available_zoom_levels": [
                "City-Wide (0.025°)",
                "Block-Level (0.01°)",
                "Zoomed-In (0.005°)"
            ],
            "available_resolutions": [
                "Coarse (10m)",
                "Standard (5m)",
                "Fine (2.5m)"
            ],
            "analysis_types": ["urban_change"],
            "max_batch_size": 20,
            "supported_image_formats": ["PNG", "JPEG", "TIFF"],
            "endpoints": {
                "simple_api_count": 3,
                "advanced_api_count": 8,
                "total_endpoints": 11
            }
        }
    }

@app.get("/locations/dates")
async def get_available_dates_endpoint(lat: float, lon: float, zoom_level: str = "City-Wide (0.025°)"):
    """
    Get available satellite imagery dates for a location
    """
    try:
        buffer = map_bbox_choice(zoom_level)
        dates = get_available_dates(lat, lon, buffer)
        return {
            "status": "success",
            "available_dates": dates,
            "total_dates": len(dates)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get available dates: {str(e)}")

@app.get("/stats/summary")
async def get_analysis_summary():
    """
    Get summary statistics of all analyses
    """
    try:
        if not analysis_history:
            return {
                "status": "success",
                "summary": {
                    "total_analyses": 0,
                    "average_change_percentage": 0,
                    "recent_analyses": 0
                }
            }

        total_analyses = len(analysis_history)
        change_percentages = [a["result"]["statistics"]["change_percentage"] for a in analysis_history]
        avg_change = sum(change_percentages) / len(change_percentages) if change_percentages else 0

        from datetime import timedelta
        recent_date = datetime.now() - timedelta(days=7)
        recent_analyses = sum(1 for a in analysis_history
                            if datetime.fromisoformat(a["timestamp"]) > recent_date)

        return {
            "status": "success",
            "summary": {
                "total_analyses": total_analyses,
                "average_change_percentage": round(avg_change, 2),
                "max_change_percentage": max(change_percentages) if change_percentages else 0,
                "min_change_percentage": min(change_percentages) if change_percentages else 0,
                "recent_analyses": recent_analyses
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")

# ===== SOCIOECONOMIC AND REAL ESTATE DATA ENDPOINTS =====

@app.get("/locations/{location}/socioeconomic")
async def get_socioeconomic_data(location: str):
    """Get comprehensive socioeconomic and real estate data for a location"""
    try:
        lat, lon = get_coordinates(location)
        if lat is None:
            raise HTTPException(status_code=404, detail="Location not found")

        comprehensive_data = data_service.get_comprehensive_analysis(lat, lon, location)

        if comprehensive_data.get("error"):
            raise HTTPException(status_code=404, detail=comprehensive_data["error"])

        return {
            "location": location,
            "coordinates": {"latitude": lat, "longitude": lon},
            "zip_code": comprehensive_data.get("zip_code"),
            "census_data": comprehensive_data.get("census_data"),
            "real_estate_data": comprehensive_data.get("real_estate_data"),
            "analysis": comprehensive_data.get("analysis")
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/zip-codes/{zip_code}/analysis")
async def get_zip_code_analysis(zip_code: str):
    """Get analysis for a specific ZIP code"""
    try:
        census_data = data_service.get_census_data(zip_code)
        real_estate_data = data_service.get_real_estate_data(zip_code)

        if not census_data or not real_estate_data:
            raise HTTPException(status_code=404, detail="No data available for this ZIP code")

        analysis = data_service._generate_analysis(census_data, real_estate_data)

        return {
            "zip_code": zip_code,
            "census_data": census_data,
            "real_estate_data": real_estate_data,
            "analysis": analysis
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/datasets/info")
async def get_dataset_info():
    """Get information about available datasets"""
    return {
        "census_data": {
            "description": "US Census demographic and socioeconomic data by ZIP code",
            "fields": [
                "population", "median_income", "poverty_rate", "unemployment_rate",
                "education_bachelor_plus", "median_age", "housing_units",
                "owner_occupied_rate", "median_home_value"
            ]
        },
        "real_estate_data": {
            "description": "Real estate market data and trends by ZIP code",
            "fields": [
                "avg_home_price", "price_per_sqft", "inventory_months", "days_on_market",
                "new_construction_permits", "foreclosure_rate", "rent_median",
                "rent_growth_yoy", "commercial_vacancy_rate"
            ]
        },
        "geographic_mapping": {
            "description": "Geographic location mappings with coordinates",
            "fields": ["city", "state", "zip_code", "latitude", "longitude", "region", "metro_area"]
        }
    }

# ===== NDVI ANALYSIS ENDPOINT =====

@app.post("/analyze/ndvi", response_model=NDVIAnalysisResponse)
async def analyze_ndvi(request: NDVIAnalysisRequest):
    """
    Comprehensive NDVI-based vegetation change detection and analysis

    This endpoint provides advanced vegetation analysis using NDVI (Normalized Difference Vegetation Index)
    calculations from Sentinel-2 satellite imagery. It includes:
    - NDVI change detection between two time periods
    - Vegetation gain/loss analysis
    - Urban expansion detection
    - Water body changes
    - Smart date selection for optimal imagery
    - Comprehensive visualizations
    - Actionable recommendations
    - Socioeconomic correlation analysis
    """
    try:
        # 1) Geocode location
        lat, lon = get_coordinates(request.location)
        if lat is None or lon is None:
            raise HTTPException(status_code=404, detail="Location not found")

        # 2) Parse analysis intent
        intents = parse_intent(request.analysis_focus)

        # 3) Map parameters
        buf = map_bbox_choice(request.zoom_level)
        res_m = map_resolution_choice(request.resolution)

        # 4) Smart date selection for NDVI analysis
        start_date, end_date, all_dates = find_smart_dates(
            lat, lon, buf, request.timeline_start, request.timeline_end
        )

        if not start_date or not end_date:
            raise HTTPException(
                status_code=404,
                detail="No suitable Sentinel-2 imagery dates available for this area/time period"
            )

        # 5) Fetch satellite imagery with all required bands
        before_img, after_img = await asyncio.gather(
            fetch_ndvi_image(lat, lon, start_date, buf, res_m),
            fetch_ndvi_image(lat, lon, end_date, buf, res_m),
        )

        if not before_img or not after_img:
            raise HTTPException(
                status_code=502,
                detail="Failed to fetch satellite imagery. Try broader zoom level or different dates."
            )

        # 6) Compute comprehensive NDVI analysis
        ndvi_analysis = compute_ndvi_analysis(before_img["raw"], after_img["raw"])

        # 7) Extract change analysis summary
        change_stats = ndvi_analysis["statistics"]
        total_change_pct = sum(
            change_stats[key]["percentage"]
            for key in ["vegetation_gain", "vegetation_loss", "urbanization", "urban_loss", "water_gain", "water_loss"]
        )

        change_analysis = {
            "total_change_percentage": round(total_change_pct, 2),
            "dominant_change": max(change_stats.items(), key=lambda x: x[1]["percentage"] if isinstance(x[1], dict) else 0)[0],
            "vegetation_change_net": change_stats["vegetation_gain"]["percentage"] - change_stats["vegetation_loss"]["percentage"],
            "urban_change_net": change_stats["urbanization"]["percentage"] - change_stats["urban_loss"]["percentage"],
            "water_change_net": change_stats["water_gain"]["percentage"] - change_stats["water_loss"]["percentage"],
            "change_intensity": "high" if total_change_pct > 10 else "moderate" if total_change_pct > 5 else "low"
        }

        # 8) Generate recommendations if requested
        recommendations = []
        if request.want_recommendations:
            recommendations = generate_ndvi_recommendations(ndvi_analysis, intents, request.location)

        # 9) Create visualizations if requested
        visualizations = {}
        if request.want_visualizations:
            visualizations = create_ndvi_visualizations(ndvi_analysis, after_img["rgb"])

        # 10) Get socioeconomic correlation analysis
        socioeconomic_correlation = None
        try:
            comprehensive_data = data_service.get_comprehensive_analysis(lat, lon, request.location)
            if not comprehensive_data.get("error"):
                # Correlate NDVI changes with socioeconomic indicators
                census_data = comprehensive_data.get("census_data", {})
                real_estate_data = comprehensive_data.get("real_estate_data", {})

                socioeconomic_correlation = {
                    "median_income": census_data.get("median_income", 0),
                    "poverty_rate": census_data.get("poverty_rate", 0),
                    "education_level": census_data.get("education_bachelor_plus", 0),
                    "home_values": real_estate_data.get("avg_home_price", 0),
                    "development_permits": real_estate_data.get("new_construction_permits", 0),
                    "correlation_insights": {
                        "development_pressure": "high" if real_estate_data.get("new_construction_permits", 0) > 20 else "low",
                        "gentrification_indicator": "high" if census_data.get("median_income", 0) > 80000 and change_stats["urbanization"]["percentage"] > 3 else "low",
                        "environmental_justice": "concern" if census_data.get("poverty_rate", 0) > 15 and change_stats["vegetation_loss"]["percentage"] > 5 else "stable"
                    }
                }
        except Exception as e:
            print(f"Socioeconomic correlation analysis failed: {e}")

        # 11) Build response
        return NDVIAnalysisResponse(
            success=True,
            location=request.location,
            coordinates={"latitude": lat, "longitude": lon},
            timeline_start=start_date,
            timeline_end=end_date,
            chosen_dates=[
                {
                    "date": start_date,
                    "cloud": next((d["cloud"] for d in all_dates if d["date"] == start_date), "N/A")
                },
                {
                    "date": end_date,
                    "cloud": next((d["cloud"] for d in all_dates if d["date"] == end_date), "N/A")
                }
            ],
            available_dates_count=len(all_dates),
            timestamp=datetime.utcnow().isoformat() + "Z",
            ndvi_analysis={
                "ndvi_statistics": ndvi_analysis["ndvi_statistics"],
                "change_statistics": change_stats,
                "analysis_focus": request.analysis_focus,
                "detected_intents": intents
            },
            change_analysis=change_analysis,
            recommendations=recommendations,
            visualizations=visualizations,
            socioeconomic_correlation=socioeconomic_correlation
        )

    except HTTPException:
        raise
    except Exception as e:
        return NDVIAnalysisResponse(
            success=False,
            location=request.location,
            coordinates={"latitude": 0, "longitude": 0},
            timeline_start="",
            timeline_end="",
            chosen_dates=[],
            available_dates_count=0,
            timestamp=datetime.utcnow().isoformat() + "Z",
            ndvi_analysis={"error": str(e)},
            change_analysis={"error": str(e)},
            recommendations=[f"Analysis failed: {str(e)}"],
            visualizations={},
            socioeconomic_correlation=None
        )

@app.get("/analyze/ndvi/{location}/quick")
async def quick_ndvi_analysis(location: str, zoom_level: str = "City-Wide (0.025°)"):
    """
    Quick NDVI analysis with default parameters for rapid assessment
    """
    request = NDVIAnalysisRequest(
        location=location,
        zoom_level=zoom_level,
        want_recommendations=True,
        want_visualizations=True,
        analysis_focus="vegetation"
    )
    return await analyze_ndvi(request)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)