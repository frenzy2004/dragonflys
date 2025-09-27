# Malaysia Location Scoring API

A FastAPI backend for calculating SME-focused location scores based on Malaysian market conditions.

## Features

- **Malaysia-specific scoring algorithm** tuned for SME realities
- **Weighted scoring system** (Competition 40%, Growth 30%, Seasonality 20%, Sentiment 10%)
- **Risk factor analysis** and opportunity identification
- **Batch processing** for multiple locations
- **Grade system** (A+ to D) with recommendations
- **CORS enabled** for frontend integration

## Quick Start

### 1. Install Dependencies
```bash
cd backend-1
pip install -r requirements.txt
```

### 2. Run the API
```bash
python main.py
```

The API will be available at:
- **Main API**: http://localhost:8001
- **Interactive docs**: http://localhost:8001/docs
- **Health check**: http://localhost:8001/health

### 3. Test the API
```bash
python test_api.py
```

## API Endpoints

### POST `/location-score`
Calculate location score for a single location.

**Request:**
```json
{
  "competition_score": 72,
  "growth_score": 81,
  "seasonality_score": 88,
  "sentiment_score": 66
}
```

**Response:**
```json
{
  "location_score": 77.9,
  "grade": "A",
  "recommendation": "Very good opportunity - Strong market conditions",
  "breakdown": {
    "scores": {
      "competition": 72,
      "growth": 81,
      "seasonality": 88,
      "sentiment": 66
    },
    "weighted_contributions": {
      "competition": 28.8,
      "growth": 24.3,
      "seasonality": 17.6,
      "sentiment": 6.6
    },
    "weights": {
      "competition": "40%",
      "growth": "30%",
      "seasonality": "20%",
      "sentiment": "10%"
    }
  },
  "risk_factors": [],
  "opportunities": [
    "Strong urban development - Growing customer base",
    "Stable seasonal demand - Predictable revenue"
  ]
}
```

### POST `/batch-score`
Calculate scores for multiple locations at once (max 50).

### GET `/score-ranges`
Get information about score ranges and their meanings.

### GET `/health`
Health check endpoint.

## Frontend Integration

### JavaScript/React Example
```javascript
const calculateLocationScore = async (scores) => {
  const response = await fetch('http://localhost:8001/location-score', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(scores)
  });
  
  return await response.json();
};

// Usage
const result = await calculateLocationScore({
  competition_score: 72,
  growth_score: 81,
  seasonality_score: 88,
  sentiment_score: 66
});

console.log(`Score: ${result.location_score} (${result.grade})`);
```

### React Component Example
```jsx
import React, { useState } from 'react';

const LocationScorer = () => {
  const [scores, setScores] = useState({
    competition_score: 50,
    growth_score: 50,
    seasonality_score: 50,
    sentiment_score: 50
  });
  const [result, setResult] = useState(null);

  const calculateScore = async () => {
    const response = await fetch('http://localhost:8001/location-score', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(scores)
    });
    const data = await response.json();
    setResult(data);
  };

  return (
    <div>
      <h2>Malaysia Location Scorer</h2>
      
      {/* Input sliders */}
      <div>
        <label>Competition (40%): {scores.competition_score}</label>
        <input 
          type="range" 
          min="0" 
          max="100" 
          value={scores.competition_score}
          onChange={(e) => setScores({...scores, competition_score: parseInt(e.target.value)})}
        />
      </div>
      
      <div>
        <label>Growth (30%): {scores.growth_score}</label>
        <input 
          type="range" 
          min="0" 
          max="100" 
          value={scores.growth_score}
          onChange={(e) => setScores({...scores, growth_score: parseInt(e.target.value)})}
        />
      </div>
      
      <div>
        <label>Seasonality (20%): {scores.seasonality_score}</label>
        <input 
          type="range" 
          min="0" 
          max="100" 
          value={scores.seasonality_score}
          onChange={(e) => setScores({...scores, seasonality_score: parseInt(e.target.value)})}
        />
      </div>
      
      <div>
        <label>Sentiment (10%): {scores.sentiment_score}</label>
        <input 
          type="range" 
          min="0" 
          max="100" 
          value={scores.sentiment_score}
          onChange={(e) => setScores({...scores, sentiment_score: parseInt(e.target.value)})}
        />
      </div>
      
      <button onClick={calculateScore}>Calculate Score</button>
      
      {result && (
        <div>
          <h3>Results</h3>
          <p><strong>Score:</strong> {result.location_score} ({result.grade})</p>
          <p><strong>Recommendation:</strong> {result.recommendation}</p>
          
          {result.opportunities.length > 0 && (
            <div>
              <h4>Opportunities:</h4>
              <ul>
                {result.opportunities.map((opp, i) => <li key={i}>{opp}</li>)}
              </ul>
            </div>
          )}
          
          {result.risk_factors.length > 0 && (
            <div>
              <h4>Risk Factors:</h4>
              <ul>
                {result.risk_factors.map((risk, i) => <li key={i}>{risk}</li>)}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default LocationScorer;
```

## Scoring Algorithm

The algorithm uses a weighted approach optimized for Malaysian SME conditions:

- **Competition (40%)**: Market saturation and rivalry intensity
- **Growth (30%)**: Urban development and expansion potential  
- **Seasonality (20%)**: Demand stability across seasons (monsoon, festivals, holidays)
- **Sentiment (10%)**: Local reviews and community perception

### Score Ranges
- **85-100 (A+)**: Excellent opportunity
- **75-84 (A)**: Very good opportunity  
- **65-74 (B+)**: Good opportunity
- **55-64 (B)**: Moderate opportunity
- **45-54 (C)**: Challenging opportunity
- **0-44 (D)**: Not recommended

## Development

### Project Structure
```
backend-1/
├── main.py           # FastAPI application
├── requirements.txt  # Dependencies
├── test_api.py      # Test script
└── README.md        # This file
```

### Running Tests
```bash
# Start the API first
python main.py

# In another terminal, run tests
python test_api.py
```

### API Documentation
Visit http://localhost:8001/docs for interactive API documentation.

## Production Deployment

For production, consider:
1. Use environment variables for configuration
2. Add authentication/rate limiting
3. Use a production ASGI server like Gunicorn
4. Configure CORS origins properly
5. Add logging and monitoring
