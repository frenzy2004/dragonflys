#!/usr/bin/env python3
"""
Test script for Malaysia Location Scoring API
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8001"

def test_single_location():
    """Test single location scoring"""
    print("🧪 Testing single location scoring...")
    
    # Test data for Cyberjaya (tech hub)
    test_data = {
        "competition_score": 72,  # High tech competition
        "growth_score": 81,       # Rapid urban development
        "seasonality_score": 88,  # Stable demand
        "sentiment_score": 66     # Good but not excellent reviews
    }
    
    response = requests.post(f"{BASE_URL}/location-score", json=test_data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Location Score: {result['location_score']}")
        print(f"📊 Grade: {result['grade']}")
        print(f"💡 Recommendation: {result['recommendation']}")
        print(f"⚠️  Risk Factors: {result['risk_factors']}")
        print(f"🚀 Opportunities: {result['opportunities']}")
        print(f"📈 Breakdown: {json.dumps(result['breakdown'], indent=2)}")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

def test_batch_locations():
    """Test batch location scoring"""
    print("\n🧪 Testing batch location scoring...")
    
    # Test data for multiple Malaysian locations
    locations = [
        {
            "competition_score": 72,
            "growth_score": 81,
            "seasonality_score": 88,
            "sentiment_score": 66
        },  # Cyberjaya
        {
            "competition_score": 85,
            "growth_score": 45,
            "seasonality_score": 70,
            "sentiment_score": 55
        },  # KL City Center (high competition)
        {
            "competition_score": 35,
            "growth_score": 92,
            "seasonality_score": 75,
            "sentiment_score": 80
        }   # Emerging suburb
    ]
    
    response = requests.post(f"{BASE_URL}/batch-score", json=locations)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Processed {result['total_processed']} locations")
        for i, location_result in enumerate(result['results']):
            print(f"Location {i+1}: Score {location_result['location_score']} (Grade: {location_result['grade']})")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

def test_score_ranges():
    """Test score ranges endpoint"""
    print("\n🧪 Testing score ranges...")
    
    response = requests.get(f"{BASE_URL}/score-ranges")
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Score Ranges:")
        for range_key, range_info in result['score_ranges'].items():
            print(f"  {range_key}: {range_info['grade']} - {range_info['meaning']}")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

def test_health_check():
    """Test health check endpoint"""
    print("\n🧪 Testing health check...")
    
    response = requests.get(f"{BASE_URL}/health")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Service Status: {result['status']}")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

def test_edge_cases():
    """Test edge cases and validation"""
    print("\n🧪 Testing edge cases...")
    
    # Test with invalid scores (out of range)
    invalid_data = {
        "competition_score": 150,  # Invalid: > 100
        "growth_score": -10,       # Invalid: < 0
        "seasonality_score": 50,
        "sentiment_score": 75
    }
    
    response = requests.post(f"{BASE_URL}/location-score", json=invalid_data)
    
    if response.status_code == 422:
        print("✅ Validation working - rejected invalid scores")
    else:
        print(f"❌ Validation failed: {response.status_code}")

def main():
    """Run all tests"""
    print("🚀 Starting Malaysia Location Scoring API Tests\n")
    
    try:
        # Check if API is running
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ API is not running. Start it with: python main.py")
            return
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to API. Make sure it's running on port 8001")
        return
    
    # Run tests
    test_health_check()
    test_single_location()
    test_batch_locations()
    test_score_ranges()
    test_edge_cases()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    main()
