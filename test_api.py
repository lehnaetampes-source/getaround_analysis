import requests

url = "http://127.0.0.1:8000/predict"

data = {
    "input": [["Renault", 80000, 120, "diesel", "black", "sedan",
               True, True, True, False, True, True, False]]
}

response = requests.post(url, json=data)
print(response.json())