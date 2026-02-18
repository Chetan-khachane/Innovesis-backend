import pandas as pd
import numpy as np
import random

np.random.seed(42)

n = 4996  # 4996 + 4 teammates = 5000

locations = ["Mumbai","Delhi","Bangalore","Hyderabad","Pune","Chennai","Kolkata","Ahmedabad"]
life_events = ["None","Marriage","New Baby","New Job","Policy Expiring"]
insurance_types = ["Health","Car","Home","Life"]

first_names = ["Rahul","Amit","Sneha","Priya","Arjun","Neha","Rohan","Ananya","Vikram","Kavya"]
last_names = ["Sharma","Patel","Singh","Verma","Iyer","Gupta","Nair","Reddy","Mehta","Joshi"]

data = []

# ---- Add teammates FIRST ----

team_members = [
    ("Chetan Khachane", 22, "chetankhachane655@gmail.com", "+918329391715"),
    ("Isha", 21, "isha.231663201@vcet.edu.in", "+919867542729"),
    ("Disha", 21, "disha.231873201@vcet.edu.in", "+918097783653"),
    ("Atharv", 22, "atharv.231673101@vcet.edu.in", "+919152274885")
]

for name, age, email, phone in team_members:
    data.append([
        name,
        age,
        email,
        phone,
        random.choice(locations),
        np.random.randint(5, 25),
        random.choice(life_events),
        random.choice(insurance_types),
        500, 300, 200, 100, 150, 80, 1
    ])

# ---- Generate rest synthetic ----

for i in range(n):
    first = random.choice(first_names)
    last = random.choice(last_names)

    age = np.random.randint(21, 65)
    income = np.random.randint(3, 40)

    data.append([
        first + " " + last,
        age,
        f"{first.lower()}.{last.lower()}{np.random.randint(1000,9999)}@gmail.com",
        "+91" + "".join([str(np.random.randint(0,9)) for _ in range(10)]),
        random.choice(locations),
        income,
        random.choice(life_events),
        random.choice(insurance_types),
        np.random.randint(10,600),
        np.random.randint(5,500),
        np.random.randint(5,600),
        np.random.randint(5,300),
        np.random.randint(5,250),
        np.random.randint(5,150),
        np.random.choice([0,1], p=[0.6,0.4])
    ])

columns = [
    "name","age","email","phone_number","location","income_lpa",
    "life_event","insurance_type",
    "whatsapp_usage_minutes_per_week",
    "facebook_usage_minutes_per_week",
    "instagram_usage_minutes_per_week",
    "telegram_usage_minutes_per_week",
    "gmail_usage_minutes_per_week",
    "sms_usage_minutes_per_week",
    "purchased"
]

df = pd.DataFrame(data, columns=columns)
df.to_csv("insurance_marketing_dataset_5000_final.csv", index=False)
