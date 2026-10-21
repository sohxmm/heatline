"""Declared regional coverage; only Nagpur-May has an assessed ANN."""

CITIES = [
    {"id": "nagpur", "name": "Nagpur", "latitude": 21.1458, "longitude": 79.0882},
    {"id": "akola", "name": "Akola", "latitude": 20.7002, "longitude": 77.0082},
    {"id": "amravati", "name": "Amravati", "latitude": 20.9320, "longitude": 77.7523},
    {"id": "chandrapur", "name": "Chandrapur", "latitude": 19.9615, "longitude": 79.2961},
    {"id": "pune", "name": "Pune", "latitude": 18.5204, "longitude": 73.8567},
    {"id": "mumbai", "name": "Mumbai", "latitude": 19.0760, "longitude": 72.8777},
]
CITY_IDS = frozenset(city["id"] for city in CITIES)
TIMEZONE = "Asia/Kolkata"
