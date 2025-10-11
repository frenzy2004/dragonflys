#!/usr/bin/env python3
"""
Download and save satellite images from the API
"""

import requests
import json
import base64
from PIL import Image
import io
import os

def save_images_for_location(location="Cyberjaya"):
    """Download and save before/after/overlay images"""

    print(f"🛰️  Getting images for: {location}")

    # Request data
    data = {
        "location": location,
        "zoom_level": "City-Wide (0.025°)",
        "resolution": "Standard (5m)",
        "alpha": 0.4
    }

    try:
        # Call the API
        response = requests.post(
            "http://localhost:8000/detect-change",
            json=data,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()

            if result.get("success"):
                print("✅ Got satellite data!")

                # Create output directory
                output_dir = f"images_{location.replace(' ', '_')}"
                os.makedirs(output_dir, exist_ok=True)

                # Save each image
                images = result.get("images", {})

                for image_type, base64_data in images.items():
                    if base64_data:
                        try:
                            # Decode base64 to image
                            image_data = base64.b64decode(base64_data)
                            image = Image.open(io.BytesIO(image_data))

                            # Save the image
                            filename = f"{output_dir}/{image_type}.png"
                            image.save(filename)
                            print(f"💾 Saved: {filename}")

                        except Exception as e:
                            print(f"❌ Error saving {image_type}: {e}")

                # Print info
                coords = result.get("coordinates", {})
                dates = result.get("dates", {})
                stats = result.get("statistics", {})

                print(f"\n📍 Location: {coords}")
                print(f"📅 Dates: {dates}")
                print(f"📊 Change: {stats.get('change_percentage', 0):.2f}%")
                print(f"📁 Images saved in: {output_dir}/")

            else:
                print(f"❌ API Error: {result.get('message')}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Test with Malaysian locations
    locations = ["Cyberjaya", "Kuala Lumpur", "Putrajaya"]

    for location in locations:
        save_images_for_location(location)
        print("-" * 50)