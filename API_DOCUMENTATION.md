# 🐉 Dragonfly API Documentation

## 📡 Unified Geospatial Change Detection API

**Version:** 2.0.0  
**Base URL:** `http://localhost:8000`  
**Interactive Docs:** `http://localhost:8000/docs`  
**Content-Type:** `application/json`

---

## 🚀 Quick Start

```python
import requests

# Simple change detection
response = requests.post("http://localhost:8000/detect-change", json={
    "location": "Cyberjaya",
    "zoom_level": "City-Wide (0.025°)",
    "resolution": "Standard (5m)",
    "alpha": 0.4
})

# NDVI Analysis
response = requests.get(
    "http://localhost:8000/analyze/ndvi/Cyberjaya/quick",
    params={"zoom_level": "City-Wide (0.025°)"}
)
```

---

## 📋 Table of Contents

1. [Authentication](#-authentication)
2. [Core Endpoints](#-core-endpoints)
3. [Advanced Analysis](#-advanced-analysis)
4. [Data Services](#-data-services)
5. [Utility Endpoints](#-utility-endpoints)
6. [Response Formats](#-response-formats)
7. [Error Handling](#-error-handling)
8. [Rate Limits](#-rate-limits)
9. [Examples](#-examples)

---

## 🔐 Authentication

Currently, the API does not require authentication. All endpoints are publicly accessible.

---

## 🎯 Core Endpoints

### Change Detection

#### `POST /detect-change`
Basic satellite change detection for any location.

**Request Body:**
```json
{
    "location": "Cyberjaya",
    "zoom_level": "City-Wide (0.025°)",
    "resolution": "Standard (5m)",
    "alpha": 0.4
}
```

**Parameters:**
- `location` (string): Location name (e.g., "Cyberjaya", "Tokyo")
- `zoom_level` (string): One of:
  - `"City-Wide (0.025°)"` - Large area overview
  - `"Block-Level (0.01°)"` - Neighborhood analysis  
  - `"Zoomed-In (0.005°)"` - Detailed building-level
- `resolution` (string): One of:
  - `"Coarse (10m)"` - Fast processing, lower detail
  - `"Standard (5m)"` - Balanced performance and quality
  - `"Fine (2.5m)"` - High detail, slower processing
- `alpha` (float, optional): Overlay transparency (0.1-1.0, default: 0.4)

**Response:**
```json
{
    "success": true,
    "message": "Change detection completed successfully",
    "coordinates": {
        "lat": 2.9339,
        "lon": 101.6456
    },
    "dates": {
        "before": "2018-03-14",
        "after": "2025-03-22"
    },
    "statistics": {
        "changed_pixels": 1234,
        "total_pixels": 65536,
        "change_percentage": 1.88
    },
    "images": {
        "before": "base64_encoded_image_data",
        "after": "base64_encoded_image_data",
        "overlay": "base64_encoded_image_data"
    }
}
```

#### `GET /detect-change/{location}/images/{image_type}`
Download individual satellite images.

**Parameters:**
- `location` (path): Location name
- `image_type` (path): One of `before`, `after`, `overlay`
- `zoom_level` (query): Zoom level
- `resolution` (query): Image resolution

**Response:** Binary image data (PNG format)

---

## 🌱 Advanced Analysis

### NDVI Analysis

#### `GET /analyze/ndvi/{location}/quick`
Quick NDVI (Normalized Difference Vegetation Index) analysis.

**Parameters:**
- `location` (path): Location name
- `zoom_level` (query): Zoom level

**Response:**
```json
{
    "status": "success",
    "location": {
        "name": "Cyberjaya",
        "coordinates": {"lat": 2.9339, "lon": 101.6456}
    },
    "ndvi_stats": {
        "mean": 0.45,
        "std": 0.12,
        "min": 0.02,
        "max": 0.89,
        "median": 0.43
    },
    "vegetation_health": "moderate",
    "change_percentage": 5.2,
    "analysis_date": "2025-09-27"
}
```

### Comprehensive Analysis

#### `POST /analyze`
Advanced geospatial analysis with comprehensive data.

**Request Body:**
```json
{
    "location": {"lat": 2.9339, "lon": 101.6456},
    "zoom_level": "City-Wide (0.025°)",
    "resolution": "Standard (5m)",
    "overlay_alpha": 0.4,
    "include_images": true
}
```

**Response:**
```json
{
    "status": "COMPLETE",
    "jobId": "uuid-string",
    "data": {
        "type": "change_detection_analysis",
        "coordinates": {"lat": 2.9339, "lon": 101.6456},
        "changePolygons": {
            "type": "FeatureCollection",
            "features": [...]
        },
        "statistics": {
            "change_percentage": 1.88,
            "area_changed": "2.5 sq km"
        },
        "images": {
            "before": "base64_data",
            "after": "base64_data",
            "overlay": "base64_data"
        }
    }
}
```

#### `POST /analyze/location`
Location-based analysis using location names.

**Request Body:**
```json
{
    "location_name": "Cyberjaya",
    "zoom_level": "Block-Level (0.01°)",
    "resolution": "Fine (2.5m)"
}
```

---

## 📊 Data Services

### Location Services

#### `GET /locations/{location}/coordinates`
Get coordinates for a location.

**Response:**
```json
{
    "location": "Cyberjaya",
    "latitude": 2.9339,
    "longitude": 101.6456
}
```

#### `POST /locations/search`
Search for locations.

**Request Body:**
```json
{
    "query": "Selangor",
    "limit": 5
}
```

**Response:**
```json
{
    "status": "success",
    "locations": [
        {
            "name": "Selangor, Malaysia",
            "display_name": "Selangor, Malaysia",
            "coordinates": {"lat": 3.2083, "lon": 101.3041},
            "place_type": "administrative",
            "country": "Malaysia"
        }
    ]
}
```

#### `GET /locations/{location}/socioeconomic`
Get socioeconomic data for a location.

**Response:**
```json
{
    "location": "Cyberjaya",
    "population": 50000,
    "income_level": "high",
    "development_index": 0.85,
    "employment_rate": 0.92
}
```

### Dataset Information

#### `GET /datasets/info`
Get information about available datasets.

**Response:**
```json
{
    "datasets": {
        "census_data": {
            "name": "Census Data",
            "description": "Population demographics",
            "records": 15,
            "last_updated": "2025-09-27"
        },
        "real_estate_data": {
            "name": "Real Estate Data",
            "description": "Property values and trends",
            "records": 30,
            "last_updated": "2025-09-27"
        }
    }
}
```

#### `GET /zip-codes/{code}/analysis`
Analyze data for a specific postal code.

**Response:**
```json
{
    "zip_code": "63000",
    "location": "Cyberjaya, Selangor",
    "analysis": {
        "population_density": "medium",
        "development_level": "high",
        "change_trend": "increasing"
    }
}
```

---

## 🔧 Utility Endpoints

### System Information

#### `GET /`
Root endpoint with API information.

**Response:**
```json
{
    "message": "Unified Geospatial Change Detection API",
    "status": "running",
    "version": "2.0.0",
    "endpoints": {
        "simple_api": {
            "detect_change": "/detect-change",
            "get_image": "/detect-change/{location}/images/{image_type}"
        },
        "advanced_api": {
            "analyze": "/analyze",
            "ndvi_analysis": "/analyze/ndvi/{location}/quick"
        }
    }
}
```

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "sentinel_hub_configured": true,
    "timestamp": "2025-09-27T11:46:31Z"
}
```

#### `GET /system/info`
Detailed system information.

**Response:**
```json
{
    "status": "success",
    "system_info": {
        "api_version": "2.0.0",
        "model_loaded": true,
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
        "max_batch_size": 20,
        "supported_image_formats": ["PNG", "JPEG", "TIFF"]
    }
}
```

### Analysis History

#### `GET /analyze/history`
Get analysis history.

**Query Parameters:**
- `limit` (int, optional): Number of results (default: 10)

**Response:**
```json
{
    "status": "success",
    "analyses": [
        {
            "id": "uuid-string",
            "timestamp": "2025-09-27T11:46:31Z",
            "location": {"lat": 2.9339, "lon": 101.6456},
            "parameters": {
                "zoom_level": "City-Wide (0.025°)",
                "resolution": "Standard (5m)"
            },
            "result": {
                "change_percentage": 1.88,
                "status": "COMPLETE"
            }
        }
    ]
}
```

#### `GET /analyze/history/{analysis_id}`
Get specific analysis by ID.

### Statistics

#### `GET /stats/summary`
Get analysis statistics summary.

**Response:**
```json
{
    "status": "success",
    "summary": {
        "total_analyses": 25,
        "average_change_percentage": 3.2,
        "max_change_percentage": 15.8,
        "min_change_percentage": 0.1,
        "recent_analyses": 5
    }
}
```

### Available Dates

#### `GET /locations/dates`
Get available satellite imagery dates for a location.

**Query Parameters:**
- `lat` (float): Latitude
- `lon` (float): Longitude
- `zoom_level` (string): Zoom level

**Response:**
```json
{
    "status": "success",
    "available_dates": [
        "2017-04-04",
        "2017-04-14",
        "2017-05-04",
        "2017-05-14"
    ],
    "total_images": 45,
    "date_range": {
        "earliest": "2017-04-04",
        "latest": "2025-03-22"
    }
}
```

---

## 📄 Response Formats

### Success Response
```json
{
    "status": "success",
    "data": { ... },
    "timestamp": "2025-09-27T11:46:31Z"
}
```

### Error Response
```json
{
    "status": "error",
    "message": "Error description",
    "error_code": "ERROR_CODE",
    "timestamp": "2025-09-27T11:46:31Z"
}
```

### Image Response
Binary data with appropriate content-type headers:
- **Content-Type:** `image/png`
- **Content-Disposition:** `attachment; filename="image.png"`

---

## ⚠️ Error Handling

### HTTP Status Codes

- `200 OK` - Request successful
- `400 Bad Request` - Invalid request parameters
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error

### Common Error Codes

- `LOCATION_NOT_FOUND` - Location could not be geocoded
- `NO_IMAGES_FOUND` - No suitable satellite images available
- `INVALID_PARAMETERS` - Request parameters are invalid
- `MODEL_ERROR` - AI model processing failed
- `SENTINEL_HUB_ERROR` - Satellite data service error

### Example Error Response
```json
{
    "status": "error",
    "message": "Not enough 0% cloud images found for this location",
    "error_code": "NO_IMAGES_FOUND",
    "timestamp": "2025-09-27T11:46:31Z"
}
```

---

## 🚦 Rate Limits

- **Default:** 100 requests per minute per IP
- **Heavy endpoints** (analysis): 10 requests per minute per IP
- **Image downloads:** 50 requests per minute per IP

Rate limit headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1632750000
```

---

## 💡 Examples

### Python Examples

#### Basic Change Detection
```python
import requests
import base64
from PIL import Image
import io

# Detect changes
response = requests.post("http://localhost:8000/detect-change", json={
    "location": "Cyberjaya",
    "zoom_level": "City-Wide (0.025°)",
    "resolution": "Standard (5m)",
    "alpha": 0.4
})

result = response.json()
if result["success"]:
    # Save images
    for img_type, base64_data in result["images"].items():
        if base64_data:
            image_data = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_data))
            image.save(f"{img_type}_image.png")
```

#### NDVI Analysis
```python
# Get NDVI analysis
response = requests.get(
    "http://localhost:8000/analyze/ndvi/Cyberjaya/quick",
    params={"zoom_level": "City-Wide (0.025°)"}
)

ndvi_data = response.json()
print(f"Vegetation Health: {ndvi_data['vegetation_health']}")
print(f"Mean NDVI: {ndvi_data['ndvi_stats']['mean']}")
```

#### Advanced Analysis
```python
# Comprehensive analysis
response = requests.post("http://localhost:8000/analyze", json={
    "location": {"lat": 2.9339, "lon": 101.6456},
    "zoom_level": "Block-Level (0.01°)",
    "resolution": "Fine (2.5m)",
    "include_images": True
})

analysis = response.json()
if analysis["status"] == "COMPLETE":
    print(f"Change Percentage: {analysis['data']['statistics']['change_percentage']}%")
```

#### Location Search
```python
# Search for locations
response = requests.post("http://localhost:8000/locations/search", json={
    "query": "Cyberjaya",
    "limit": 3
})

locations = response.json()
for location in locations["locations"]:
    print(f"{location['display_name']}: {location['coordinates']}")
```

### cURL Examples

#### Basic Change Detection
```bash
curl -X POST "http://localhost:8000/detect-change" \
     -H "Content-Type: application/json" \
     -d '{
       "location": "Cyberjaya",
       "zoom_level": "City-Wide (0.025°)",
       "resolution": "Standard (5m)",
       "alpha": 0.4
     }'
```

#### Download Image
```bash
curl -X GET "http://localhost:8000/detect-change/Cyberjaya/images/before?zoom_level=City-Wide%20(0.025°)&resolution=Standard%20(5m)" \
     -o before_image.png
```

#### NDVI Analysis
```bash
curl -X GET "http://localhost:8000/analyze/ndvi/Cyberjaya/quick?zoom_level=City-Wide%20(0.025°)"
```

---

## 🔗 Related Resources

- **Interactive API Documentation:** http://localhost:8000/docs
- **OpenAPI Specification:** http://localhost:8000/openapi.json
- **GitHub Repository:** https://github.com/frenzy2004/dragonfly
- **Sentinel Hub:** https://www.sentinel-hub.com/

---

**Last Updated:** September 27, 2025  
**API Version:** 2.0.0