import json

with open('house_price.json', 'r', encoding='utf-8') as f:
    json_data = json.loads(f.read())

city = "武汉"
area = "东西湖"

x = 10

k, b, min_, ave_, max_ = json_data[city][area]
print(k, b, min_, ave_, max_)
print(k * x + b)
