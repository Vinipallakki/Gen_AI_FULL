# import requests

# # Bangalore coordinates
# url = "https://api.open-meteo.com/v1/forecast?latitude=12.97&longitude=77.59&current_weather=true"

# response = requests.get(url)
# data = response.json()

# print("Temperature:", data["current_weather"]["temperature"], "°C")
# print("Wind Speed:", data["current_weather"]["windspeed"])

# import requests

# # Step 1: Get city from user
# city = input("Enter city name: ")

# # Step 2: Convert city → latitude & longitude
# geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"

# geo_response = requests.get(geo_url)
# geo_data = geo_response.json()

# # Check if city found
# if "results" not in geo_data:
#     print("❌ City not found")
# else:
#     lat = geo_data["results"][0]["latitude"]
#     lon = geo_data["results"][0]["longitude"]

#     # Step 3: Get weather using lat/lon
#     weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"

#     weather_response = requests.get(weather_url)
#     weather_data = weather_response.json()

#     # Step 4: Print result
#     print(f"\n📍 City: {geo_data['results'][0]['name']}")
#     print(f"🌡️ Temperature: {weather_data['current_weather']['temperature']}°C")
#     print(f"💨 Wind Speed: {weather_data['current_weather']['windspeed']}")

# import requests

# # Your API Key
# API_KEY = "51d20a4dbb924b2a89c101759262603"

# # Get city from user
# city = input("Enter city name: ")

# # API URL
# url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}"

# # Send GET request
# response = requests.get(url)

# # Convert to JSON
# data = response.json()

# # Debug (optional)
# # print(data)

# # Check success
# if response.status_code == 200:
#     print(f"\n📍 City: {data['location']['name']}")
#     print(f"🌡️ Temperature: {data['current']['temp_c']}°C")
#     print(f"☁️ Condition: {data['current']['condition']['text']}")
#     print(f"💨 Wind: {data['current']['wind_kph']} kph")
# else:
#     print("❌ Error:", data.get("error", {}).get("message"))