import csv
import utm
import numpy as np


X = []
Y = []
max_longitude = 0
max_latitude = 0
min_latitude = 9999999999
min_longitude = 999999999
with open('../data/train_2g.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        utm_coordinate = utm.from_latlon(float(row['Latitude']), float(row['Longitude']))
        max_latitude = max(max_latitude, utm_coordinate[0])
        max_longitude = max(max_longitude, utm_coordinate[1])
        min_latitude = min(min_latitude, utm_coordinate[0])
        min_longitude = min(min_longitude, utm_coordinate[1])


print(max_latitude - min_latitude)
print(max_longitude - min_longitude)



