#!/usr/bin/env python3
"""
Test script for Bepo API endpoints
Demonstrates storing and searching memories
"""
import requests
import json
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

def test_root():
    """Test the root endpoint"""
    print("=== Testing Root Endpoint ===")
    response = requests.get(f"{BASE_URL}/")
    print(json.dumps(response.json(), indent=2))
    print()

def test_memory(image_path, note, lat, lon):
    """Test storing a memory"""
    print(f"=== Storing Memory: {note} ===")
    with open(image_path, 'rb') as f:
        files = {'photo': f}
        data = {
            'note': note,
            'lat': lat,
            'lon': lon
        }
        response = requests.post(f"{BASE_URL}/memory", files=files, data=data)
    
    result = response.json()
    print(json.dumps(result, indent=2))
    print()
    return result

def test_search(query):
    """Test searching memories"""
    print(f"=== Searching for: '{query}' ===")
    data = {'query': query}
    response = requests.post(f"{BASE_URL}/search", data=data)
    
    result = response.json()
    print(json.dumps(result, indent=2))
    print()
    return result

def main():
    """Run all tests"""
    print("Bepo API Test Suite\n")
    print("=" * 60)
    print()
    
    # Test root endpoint
    test_root()
    
    # Note: Replace these with actual image paths to test
    # Example usage:
    # test_memory('path/to/sunset.jpg', 'Beautiful sunset at the beach', 34.0522, -118.2437)
    # test_memory('path/to/mountains.jpg', 'Snowy mountain peaks', 45.8325, 6.8652)
    
    # Example searches:
    # test_search('sunset')
    # test_search('mountains')
    # test_search('beach')
    
    print("=" * 60)
    print("\nTo test with actual images, update the main() function")
    print("with paths to your test images.")

if __name__ == "__main__":
    main()
