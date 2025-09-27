from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List
import uvicorn

app = FastAPI(
    title="Malaysia Location Scoring API",
    description="SME-focused location scoring system for Malaysian market conditions",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ScoreInput(BaseModel):
    competition_score: float = Field(..., ge=0, le=100, description="Competition intensity (0-100)")
    growth_score: float = Field(..., ge=0, le=100, description="Urban growth potential (0-100)")
    seasonality_score: float = Field(..., ge=0, le=100, description="Seasonal demand stability (0-100)")
    sentiment_score: float = Field(..., ge=0, le=100, description="Local sentiment & reviews (0-100)")

class ScoreResponse(BaseModel):
    location_score: float
    grade: str
    recommendation: str
    breakdown: Dict[str, Any]
    risk_factors: list
    opportunities: list

def location_score(competition_score, growth_score, seasonality_score, sentiment_score):
    """
    Malaysia-specific Location Score (0–100).
    Tuned for SME realities: high failure rate, urbanization, festivals, monsoon.
    """
    weights = {
        "competition": 0.4,   # 60% SME failure due to heavy competition
        "growth": 0.3,        # Is the city expanding? NDVI/urbanization
        "seasonality": 0.2,   # School holidays, Ramadan, monsoon demand shifts
        "sentiment": 0.1      # Local reviews (service, parking, halal)
    }

    score = (
        competition_score * weights["competition"] +
        growth_score * weights["growth"] +
        seasonality_score * weights["seasonality"] +
        sentiment_score * weights["sentiment"]
    )

    return round(score, 1)

def get_grade_and_recommendation(score: float) -> tuple:
    """Get letter grade and recommendation based on score"""
    if score >= 85:
        return "A+", "Excellent opportunity - High potential for success"
    elif score >= 75:
        return "A", "Very good opportunity - Strong market conditions"
    elif score >= 65:
        return "B+", "Good opportunity - Favorable conditions with some challenges"
    elif score >= 55:
        return "B", "Moderate opportunity - Requires careful planning"
    elif score >= 45:
        return "C", "Challenging opportunity - High risk, consider alternatives"
    else:
        return "D", "Not recommended - Very high risk of failure"

def analyze_risk_factors(data: ScoreInput) -> list:
    """Identify key risk factors based on scores"""
    risks = []
    
    if data.competition_score > 80:
        risks.append("High competition - Market saturation risk")
    if data.growth_score < 40:
        risks.append("Limited growth potential - Declining area")
    if data.seasonality_score < 50:
        risks.append("High seasonal volatility - Unstable demand")
    if data.sentiment_score < 40:
        risks.append("Poor local sentiment - Customer acquisition challenges")
    
    return risks

def identify_opportunities(data: ScoreInput) -> list:
    """Identify opportunities based on scores"""
    opportunities = []
    
    if data.growth_score > 70:
        opportunities.append("Strong urban development - Growing customer base")
    if data.seasonality_score > 75:
        opportunities.append("Stable seasonal demand - Predictable revenue")
    if data.sentiment_score > 70:
        opportunities.append("Positive local sentiment - Good brand potential")
    if data.competition_score < 50:
        opportunities.append("Low competition - Market gap opportunity")
    
    return opportunities

@app.get("/")
async def root():
    return {
        "message": "Malaysia Location Scoring API",
        "status": "running",
        "endpoints": {
            "score": "/location-score",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "location-scoring"}

@app.post("/location-score", response_model=ScoreResponse)
async def calculate_location_score(data: ScoreInput):
    """
    Calculate Malaysia-specific location score for SME business opportunities.
    
    Takes into account:
    - Competition intensity (40% weight)
    - Urban growth potential (30% weight) 
    - Seasonal demand patterns (20% weight)
    - Local sentiment & reviews (10% weight)
    """
    try:
        # Calculate the main score
        score = location_score(
            data.competition_score,
            data.growth_score, 
            data.seasonality_score,
            data.sentiment_score
        )
        
        # Get grade and recommendation
        grade, recommendation = get_grade_and_recommendation(score)
        
        # Analyze risks and opportunities
        risks = analyze_risk_factors(data)
        opportunities = identify_opportunities(data)
        
        # Create detailed breakdown
        breakdown = {
            "scores": {
                "competition": data.competition_score,
                "growth": data.growth_score,
                "seasonality": data.seasonality_score,
                "sentiment": data.sentiment_score
            },
            "weighted_contributions": {
                "competition": round(data.competition_score * 0.4, 1),
                "growth": round(data.growth_score * 0.3, 1),
                "seasonality": round(data.seasonality_score * 0.2, 1),
                "sentiment": round(data.sentiment_score * 0.1, 1)
            },
            "weights": {
                "competition": "40%",
                "growth": "30%", 
                "seasonality": "20%",
                "sentiment": "10%"
            }
        }
        
        return ScoreResponse(
            location_score=score,
            grade=grade,
            recommendation=recommendation,
            breakdown=breakdown,
            risk_factors=risks,
            opportunities=opportunities
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calculation error: {str(e)}")

@app.get("/score-ranges")
async def get_score_ranges():
    """Get information about score ranges and their meanings"""
    return {
        "score_ranges": {
            "85-100": {"grade": "A+", "meaning": "Excellent opportunity"},
            "75-84": {"grade": "A", "meaning": "Very good opportunity"},
            "65-74": {"grade": "B+", "meaning": "Good opportunity"},
            "55-64": {"grade": "B", "meaning": "Moderate opportunity"},
            "45-54": {"grade": "C", "meaning": "Challenging opportunity"},
            "0-44": {"grade": "D", "meaning": "Not recommended"}
        },
        "weights": {
            "competition": "40% - Market saturation and rivalry",
            "growth": "30% - Urban development and expansion",
            "seasonality": "20% - Demand stability across seasons",
            "sentiment": "10% - Local reviews and perception"
        }
    }

@app.post("/batch-score")
async def calculate_batch_scores(locations: List[ScoreInput]):
    """Calculate scores for multiple locations at once"""
    if len(locations) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 locations per batch")
    
    results = []
    for i, location_data in enumerate(locations):
        try:
            score = location_score(
                location_data.competition_score,
                location_data.growth_score,
                location_data.seasonality_score,
                location_data.sentiment_score
            )
            grade, recommendation = get_grade_and_recommendation(score)
            
            results.append({
                "index": i,
                "location_score": score,
                "grade": grade,
                "recommendation": recommendation,
                "input_data": location_data.dict()
            })
        except Exception as e:
            results.append({
                "index": i,
                "error": str(e),
                "input_data": location_data.dict()
            })
    
    return {"results": results, "total_processed": len(results)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
