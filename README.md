# 🐉 Dragonfly - Advanced Satellite Change Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20+-orange.svg)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116+-green.svg)](https://fastapi.tiangolo.com)
[![Gradio](https://img.shields.io/badge/Gradio-5.46+-purple.svg)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🌍 Overview

Dragonfly is a comprehensive satellite-based urban change detection system that combines deep learning with Sentinel-2 satellite imagery to analyze environmental and urban changes over time. The system features multiple interfaces including a web UI, REST API, and advanced geospatial analysis capabilities.

**🚀 NEW: Now featuring our custom Infinity World Model** - A cutting-edge AI system for advanced geospatial intelligence and location scoring optimized for Malaysian market conditions.

### 🔬 Key Features

- **🛰️ Satellite Imagery Analysis**: Uses Sentinel-2 data with cloud-free filtering
- **🌱 NDVI Analysis**: Normalized Difference Vegetation Index calculations
- **📊 Geospatial Intelligence**: Advanced location search and analysis
- **💾 Data Integration**: Census, real estate, and socioeconomic data
- **⚡ Real-time Processing**: Fast change detection with caching
- **🧠 Infinity World Model**: Custom AI for Malaysia-specific location scoring
- **🏢 SME Business Intelligence**: Location scoring for Malaysian small businesses

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git
- **Sentinel Hub account** (free tier available) - Required for satellite imagery access

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/frenzy2004/dragonfly.git
cd dragonfly
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure Sentinel Hub API credentials**
```bash
# Create .env file (optional - defaults are included for demo)
echo "CLIENT_ID=your_sentinel_hub_client_id" > .env
echo "CLIENT_SECRET=your_sentinel_hub_client_secret" >> .env
```

**About Sentinel Hub API:**
- **What it does**: Provides access to Sentinel-2 satellite imagery from European Space Agency
- **Why we need it**: Downloads high-resolution satellite images for change detection analysis
- **Free tier**: 1,000 requests/month, perfect for testing and small projects
- **Sign up**: [Sentinel Hub Dashboard](https://apps.sentinel-hub.com/dashboard/)
- **Default credentials**: Demo credentials are included, but get your own for production use

4. **Launch the applications**

**Web Interface (Gradio)**
```bash
python app.py
# Opens at http://127.0.0.1:7860
```

**REST API (FastAPI)**
```bash
python unified_api.py
# API available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

**Malaysia Location Scoring API (Infinity World Model)**
```bash
cd backend-1
python main.py
# API available at http://localhost:8001
# Interactive docs at http://localhost:8001/docs
```

## 🎯 Usage Examples

### Web Interface
1. Open http://127.0.0.1:7860
2. Enter a location (e.g., "Cyberjaya", "Tokyo", "London")
3. Select zoom level and resolution
4. Click "Detect Change" to analyze satellite imagery

### REST API

**Simple Change Detection**
```python
import requests

response = requests.post("http://localhost:8000/detect-change", json={
    "location": "Cyberjaya",
    "zoom_level": "City-Wide (0.025°)",
    "resolution": "Standard (5m)",
    "alpha": 0.4
})
result = response.json()
```

**NDVI Analysis**
```python
response = requests.get(
    "http://localhost:8000/analyze/ndvi/Cyberjaya/quick",
    params={"zoom_level": "City-Wide (0.025°)"}
)
ndvi_data = response.json()
```

**Advanced Geospatial Analysis**
```python
response = requests.post("http://localhost:8000/analyze", json={
    "location": {"lat": 2.9339, "lon": 101.6456},
    "zoom_level": "Block-Level (0.01°)",
    "resolution": "Fine (2.5m)",
    "include_images": True
})
```

## 🏗️ Architecture

```
Dragonfly/
├── 🌐 Web Interface (app.py)
├── 🔧 Unified API (unified_api.py)
├── 🧠 Infinity World Model (Infinity_World_Model.ipynb) - CUSTOM AI
├── 🏢 Malaysia Location Scoring API (backend-1/)
│   ├── main.py - FastAPI backend
│   ├── requirements.txt
│   ├── test_api.py
│   └── README.md
├── 📊 Data Services
│   ├── census_data.csv
│   ├── real_estate_data.csv
│   └── geographic_mapping.csv
├── 🧪 Testing
│   ├── test_unified_api.py
│   ├── test_datasets_api.py
│   └── test_change_api.py
└── 📚 Documentation
    ├── API_DOCUMENTATION.md
    └── README.md
```

## 🔬 Technical Details

### Infinity World Model (Custom AI)
- **🧠 Advanced Geospatial Intelligence**: Custom-trained model for Malaysian market analysis
- **📍 Location Scoring Algorithm**: SME-optimized scoring with Malaysia-specific weights
- **🏢 Business Intelligence**: Competition, growth, seasonality, and sentiment analysis
- **⚡ Real-time API**: FastAPI backend with comprehensive scoring endpoints
- **📊 Multi-factor Analysis**: 40% competition, 30% growth, 20% seasonality, 10% sentiment

### Satellite Change Detection Model
- **Input**: Concatenated before/after satellite images (RGB, 128x128)
- **Output**: Binary change mask highlighting detected changes
- **Framework**: TensorFlow/Keras U-Net Architecture

### Satellite Data (via Sentinel Hub API)
- **Source**: Sentinel-2 (European Space Agency)
- **API Provider**: Sentinel Hub - Professional satellite data service
- **Bands**: B04 (Red), B03 (Green), B02 (Blue)
- **Resolutions**: 10m, 5m, 2.5m
- **Cloud Filter**: Only 0% cloud coverage images used
- **Coverage**: Global coverage with 5-day revisit time
- **Data Format**: GeoTIFF with automatic preprocessing

### API Endpoints

#### Simple API
- `POST /detect-change` - Basic change detection
- `GET /detect-change/{location}/images/{type}` - Download images
- `GET /locations/{location}/coordinates` - Get coordinates

#### Advanced API
- `POST /analyze` - Comprehensive analysis
- `POST /analyze/location` - Location-based analysis
- `GET /analyze/ndvi/{location}/quick` - NDVI analysis
- `GET /locations/search` - Location search
- `GET /system/info` - System information

#### Data API
- `GET /datasets/info` - Available datasets
- `GET /locations/{location}/socioeconomic` - Socioeconomic data
- `GET /zip-codes/{code}/analysis` - Postal code analysis

#### Infinity World Model API (Port 8001)
- `POST /location-score` - Malaysia-specific location scoring
- `POST /batch-score` - Batch location analysis
- `GET /score-ranges` - Score interpretation guide
- `GET /health` - API health check

## 🌱 NDVI Analysis

The system includes advanced NDVI (Normalized Difference Vegetation Index) analysis for vegetation monitoring:

```python
# Quick NDVI analysis
GET /analyze/ndvi/{location}/quick?zoom_level=City-Wide%20(0.025°)

# Response includes:
{
  "ndvi_stats": {
    "mean": 0.45,
    "std": 0.12,
    "min": 0.02,
    "max": 0.89
  },
  "vegetation_health": "moderate",
  "change_percentage": 5.2
}
```

## 🧠 Infinity World Model - Custom AI System

### Malaysia-Specific Location Intelligence
Our custom **Infinity World Model** (`Infinity_World_Model.ipynb`) provides advanced geospatial intelligence specifically tuned for Malaysian market conditions and SME business requirements.

#### Key Capabilities
- **🏢 SME Business Scoring**: Optimized for Malaysian small business success factors
- **📊 Multi-Factor Analysis**: Competition (40%), Growth (30%), Seasonality (20%), Sentiment (10%)
- **🌏 Malaysian Market Focus**: Accounts for monsoon seasons, festivals, urbanization patterns
- **⚡ Real-time API**: FastAPI backend for instant location scoring
- **📈 Risk Assessment**: Identifies opportunities and risk factors

#### Usage Example
```python
# Malaysia Location Scoring API
import requests

response = requests.post("http://localhost:8001/location-score", json={
    "competition_score": 72,    # Market competition intensity
    "growth_score": 81,         # Urban development potential
    "seasonality_score": 88,    # Seasonal demand stability
    "sentiment_score": 66       # Local sentiment & reviews
})

result = response.json()
print(f"Location Score: {result['location_score']} ({result['grade']})")
print(f"Recommendation: {result['recommendation']}")
```

#### Score Interpretation
- **85-100 (A+)**: Excellent opportunity - High potential for success
- **75-84 (A)**: Very good opportunity - Strong market conditions
- **65-74 (B+)**: Good opportunity - Favorable conditions
- **55-64 (B)**: Moderate opportunity - Requires careful planning
- **45-54 (C)**: Challenging opportunity - High risk
- **0-44 (D)**: Not recommended - Very high risk
## 📊 Data Integration

### Available Datasets
- **Census Data**: Population demographics and statistics
- **Real Estate Data**: Property values and market trends
- **Geographic Mapping**: Location coordinates and boundaries
- **Parliamentary Data**: Administrative regions (Malaysia)

### Socioeconomic Analysis
```python
GET /locations/Cyberjaya/socioeconomic
# Returns demographic and economic indicators
```

## 🧪 Testing

Run comprehensive tests:

```bash
# Test unified API
python test_unified_api.py

# Test datasets API
python test_datasets_api.py

# Test change detection API
python test_change_api.py
```

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run individual services
cd geospatial-agent
docker build -t dragonfly-api .
docker run -p 8000:8000 dragonfly-api
```

## 📈 Performance

- **Model Inference**: ~1-2 seconds per analysis
- **Image Processing**: Optimized with caching
- **API Response**: Sub-second for cached results
- **Concurrent Users**: Supports multiple simultaneous requests

## 🔧 Configuration

### Environment Variables
```bash
CLIENT_ID=your_sentinel_hub_client_id
CLIENT_SECRET=your_sentinel_hub_client_secret
```

### Zoom Levels
- **City-Wide (0.025°)**: Large area overview
- **Block-Level (0.01°)**: Neighborhood analysis
- **Zoomed-In (0.005°)**: Detailed building-level

### Image Resolutions
- **Coarse (10m)**: Fast processing, lower detail
- **Standard (5m)**: Balanced performance and quality
- **Fine (2.5m)**: High detail, slower processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **European Space Agency** for Sentinel-2 satellite imagery
- **Sentinel Hub** for data access infrastructure
- **TensorFlow/Keras** for deep learning framework
- **FastAPI** for high-performance API framework
- **Gradio** for interactive web interface

## Support

- **Issues**: [GitHub Issues](https://github.com/frenzy2004/dragonfly/issues)
- **Discussions**: [GitHub Discussions](https://github.com/frenzy2004/dragonfly/discussions)
- **Documentation**: [API Docs](http://localhost:8000/docs) (when running)

## 🌟 Features Roadmap

- [x] **Infinity World Model** - Custom AI for Malaysian market analysis
- [x] **Malaysia Location Scoring API** - SME-focused business intelligence
- [ ] Real-time change monitoring
- [ ] Mobile app integration
- [ ] Machine learning model improvements
- [ ] Additional satellite data sources
- [ ] Cloud deployment templates
- [ ] Advanced visualization tools
- [ ] Infinity World Model v2.0 - Enhanced AI capabilities

---

**Made with ❤️ for environmental monitoring, urban development analysis, and Malaysian SME business intelligence**

*Featuring our custom Infinity World Model - Advanced AI for geospatial intelligence*

*Last updated: September 2025*
