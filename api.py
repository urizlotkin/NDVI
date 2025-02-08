import json 
import requests
import pandas as pd

df = pd.read_csv('data/data points.csv')
url = "https://api.ims.gov.il/v1/Envista/stations"
headers = { 'Authorization': 'ApiToken f058958a-d8bd-47cc-95d7-7ecf98610e47' } 
response = requests.request("GET", url, headers=headers) 
data = json.loads(response.text.encode('utf8'))
names = []
for d in data:
    names.append(d['name'].lower())
    # print(d['name'])

station_names = df['Station_na'].str.lower().tolist()

for name in names:
    if name in station_names:
        print(name)
        print('find!!~!!')


# print(names)