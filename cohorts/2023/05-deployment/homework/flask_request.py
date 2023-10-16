import requests

url = 'http://127.0.0.1:9696/predict'
# client = {'job': 'unknown', 'duration': 270, 'poutcome': 'failure'}
client = {'job': 'retired', 'duration': 445, 'poutcome': 'success'}

result  = requests.post(url, json=client, timeout=10).json()
print(result)
