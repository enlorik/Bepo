#!/usr/bin/env python3
"""
Example script demonstrating the Bepo API
Creates test images and shows how to store and search memories
"""
import os
import sys
import requests
import json
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"
TEMP_DIR = "/tmp/bepo_examples"

def create_test_images():
    """Create sample test images"""
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Create a sunset image
    print("Creating sunset image...")
    sunset = Image.new('RGB', (400, 300), color='#87CEEB')
    draw = ImageDraw.Draw(sunset)
    # Sun
    draw.ellipse([150, 80, 250, 180], fill='#FFD700', outline='#FFA500', width=3)
    # Ocean
    draw.rectangle([0, 200, 400, 300], fill='#1E90FF')
    # Beach
    draw.rectangle([0, 220, 400, 300], fill='#F4A460')
    sunset_path = os.path.join(TEMP_DIR, 'sunset.jpg')
    sunset.save(sunset_path)
    
    # Create a mountain image
    print("Creating mountain image...")
    mountain = Image.new('RGB', (400, 300), color='#87CEEB')
    draw = ImageDraw.Draw(mountain)
    # Mountains
    draw.polygon([(0, 200), (100, 100), (200, 200)], fill='#808080', outline='#696969')
    draw.polygon([(150, 200), (250, 80), (350, 200)], fill='#696969', outline='#000000')
    draw.polygon([(300, 200), (400, 120), (400, 200)], fill='#808080', outline='#696969')
    # Snow on peaks
    draw.polygon([(250, 80), (220, 120), (280, 120)], fill='white')
    # Ground
    draw.rectangle([0, 200, 400, 300], fill='#228B22')
    mountain_path = os.path.join(TEMP_DIR, 'mountain.jpg')
    mountain.save(mountain_path)
    
    # Create a city image
    print("Creating city image...")
    city = Image.new('RGB', (400, 300), color='#87CEEB')
    draw = ImageDraw.Draw(city)
    # Buildings
    draw.rectangle([50, 150, 100, 280], fill='#708090', outline='#000000')
    draw.rectangle([120, 120, 180, 280], fill='#778899', outline='#000000')
    draw.rectangle([200, 140, 250, 280], fill='#708090', outline='#000000')
    draw.rectangle([270, 100, 330, 280], fill='#778899', outline='#000000')
    # Windows
    for x in range(60, 95, 15):
        for y in range(160, 275, 20):
            draw.rectangle([x, y, x+8, y+12], fill='#FFFF00')
    # Ground
    draw.rectangle([0, 280, 400, 300], fill='#696969')
    city_path = os.path.join(TEMP_DIR, 'city.jpg')
    city.save(city_path)
    
    return {
        'sunset': sunset_path,
        'mountain': mountain_path,
        'city': city_path
    }

def check_server():
    """Check if the server is running"""
    try:
        response = requests.get(f"{BASE_URL}/", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def store_memory(image_path, note, lat, lon):
    """Store a memory"""
    print(f"\n=== Storing: {note} ===")
    with open(image_path, 'rb') as f:
        files = {'photo': f}
        data = {
            'note': note,
            'lat': lat,
            'lon': lon
        }
        response = requests.post(f"{BASE_URL}/memory", files=files, data=data)
    
    result = response.json()
    print(f"✓ Memory stored with ID: {result.get('memory_id')}")
    print(f"  Image: {result.get('image_path')}")
    print(f"  Location: ({result.get('lat')}, {result.get('lon')})")
    return result

def search_memory(query):
    """Search for memories"""
    print(f"\n=== Searching for: '{query}' ===")
    data = {'query': query}
    response = requests.post(f"{BASE_URL}/search", data=data)
    
    result = response.json()
    if result['status'] == 'success' and 'match' in result:
        match = result['match']
        print(f"✓ Found match (score: {match['score']:.4f}):")
        print(f"  Note: {match['note']}")
        print(f"  Location: ({match['lat']}, {match['lon']})")
        print(f"  Image: {match['image_path']}")
    else:
        print("✗ No matches found")
    return result

def main():
    """Run the example workflow"""
    print("=" * 70)
    print("Bepo API Example - Memory Storage and Search")
    print("=" * 70)
    
    # Check if server is running
    if not check_server():
        print("\n❌ Error: Server is not running!")
        print("Please start the server first with: python main.py")
        sys.exit(1)
    
    print("\n✓ Server is running")
    
    # Create test images
    print("\n--- Creating Test Images ---")
    images = create_test_images()
    print(f"✓ Created {len(images)} test images in {TEMP_DIR}")
    
    # Store memories
    print("\n--- Storing Memories ---")
    store_memory(
        images['sunset'],
        'Beautiful sunset at the beach',
        34.0522,  # Los Angeles, CA
        -118.2437
    )
    
    store_memory(
        images['mountain'],
        'Snowy mountain peaks in winter',
        45.8325,  # Mont Blanc, France
        6.8652
    )
    
    store_memory(
        images['city'],
        'City skyline at night with lights',
        40.7128,  # New York City, NY
        -74.0060
    )
    
    # Search for memories
    print("\n--- Searching Memories ---")
    search_memory('sunset')
    search_memory('mountains')
    search_memory('city')
    search_memory('beach')
    search_memory('winter')
    search_memory('lights')
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
