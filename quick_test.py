#!/usr/bin/env python3
"""
Quick test for cloud coverage changes
Tests if the 45% cloud threshold is working for Malaysian locations
"""

import requests
import json
import time

def test_cloud_coverage():
    """Test the updated cloud coverage system"""

    # Malaysian test locations (should now work better!)
    locations = [
        "Cyberjaya",
        "Kuala Lumpur",
        "Johor Bahru",
        "Putrajaya"
    ]

    # Test data
    test_data = {
        "zoom_level": "City-Wide (0.025°)",
        "resolution": "Standard (5m)",
        "alpha": 0.4
    }

    print("🇲🇾 Testing Malaysian Locations with 45% Cloud Coverage...")
    print("=" * 60)

    for location in locations:
        print(f"\n🛰️  Testing: {location}")
        print("-" * 30)

        test_data["location"] = location

        try:
            # Test the API endpoint
            response = requests.post(
                "http://localhost:8000/detect-change",
                json=test_data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    print(f"✅ SUCCESS: Found satellite images!")
                    print(f"   Coordinates: {result.get('coordinates', {})}")
                    print(f"   Dates: {result.get('dates', {})}")
                    stats = result.get('statistics', {})
                    if stats:
                        print(f"   Change: {stats.get('change_percentage', 0):.2f}%")
                else:
                    print(f"❌ FAILED: {result.get('message', 'Unknown error')}")
            else:
                print(f"❌ HTTP ERROR: {response.status_code}")
                print(f"   Response: {response.text[:200]}")

        except requests.exceptions.ConnectionError:
            print("❌ CONNECTION ERROR: Make sure the API is running!")
            print("   Run: python unified_api.py")
            break
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")

        time.sleep(1)  # Be nice to the API

    print("\n" + "=" * 60)
    print("🧪 Test Complete!")
    print("\nIf you see ✅ SUCCESS messages, the 45% cloud coverage is working!")
    print("If you see ❌ FAILED messages, the old 0% limit might still be there.")

if __name__ == "__main__":
    test_cloud_coverage()