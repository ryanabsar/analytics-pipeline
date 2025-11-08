import csv
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from faker import Faker
import datetime
from collections import defaultdict
import logging
import random


fake = Faker()

datapath = "data_generation/data"

# Utility functions

def random_date(start, end):
    """Generate a random datetime between `start` and `end`"""
    return start + timedelta(
        seconds=np.random.randint(0, int((end - start).total_seconds()))
    )

def weighted_choice(choices, weights):
    """Return a random element from choices based on the provided weights."""
    return np.random.choice(choices, p=np.array(weights) / sum(weights))

# read product data
def generate_products() -> List[dict]:
    """Load products from CSV and return as a list of dictionaries."""
    df = pd.read_csv(f'{datapath}/products.csv')
    products = df.to_dict(orient='list')

    product_brand_dict = defaultdict(list)
    product_category_dict = defaultdict(list)
    gender_category_dict = defaultdict(list)
    product_gender_dict = defaultdict(list)
    product_by_id_dict = {}

    for _, row in df.iterrows():
        pid = row['id']
        brand = row['brand']
        name = row['name']
        cost = row['cost']
        category = row['category']
        department = row['department']
        sku = row['sku']
        retail_price = row['retail_price']
        dc_id = row['distribution_center_id']

        product_tuple = (pid, brand, name, cost, category, department, sku, retail_price, dc_id) 

        # by id

        product_by_id_dict[pid] = {
            "brand": brand,
            "name": name,
            "cost": cost,
            "category": category,
            "department": department,
            "sku": sku,
            "retail_price": retail_price,
            "distribution_center_id": dc_id
        }

        # by brand and category
        product_brand_dict[brand].append(product_tuple)
        product_category_dict[category].append(product_tuple)

        # group by gender
        gender_key = "M" if department == "Men" else "F" if department == "Women" else None

        if gender_key:
            product_gender_dict[gender_key].append(product_tuple)
            gender_category_dict[gender_key + category].append(product_tuple)

    return dict(product_gender_dict),product_by_id_dict, products


def generate_locations() -> List[str]:
    """Load world population from CSV and return as a list."""
    df = pd.read_csv(f'{datapath}/world_pop.csv')
    return df

SECONDS_IN_MINUTE = 60
MINUTES_IN_HOUR = 60
MINUTES_IN_DAY = 1440
MIN_AGE = 12
MAX_AGE = 71

products = generate_products()
LOCATION_DATA  = generate_locations()
PRODUCT_GENDER_DICT , PRODUCT_BY_ID_DICT , PRODUCT_LIST = products
logging.info(f"Loaded {len(PRODUCT_BY_ID_DICT)} products.")
logging.info(f"Loaded {len(LOCATION_DATA)} world population records.")


def get_address() -> dict:
    """Generate a random address using Faker using LOCATION_DATA weighted by population."""
    chosen = LOCATION_DATA.sample(weights = LOCATION_DATA['population'], n=1).iloc[0]

    return dict(
        street=fake.street_address(),
        city=chosen["city"],
        state=chosen["state"],
        postal_code=str(chosen["postal_code"]),
        country=chosen["country"],
        latitude=float(chosen["latitude"]),
        longitude=float(chosen["longitude"]),
    )


# generates random date between now and specified date
def created_at(start_date: datetime.datetime) -> datetime.datetime:
    end_date = datetime.datetime.now()
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    if days_between_dates <= 1:
        days_between_dates = 2
    random_number_of_days = random.randrange(1, days_between_dates)
    created_at = (
        start_date
        + datetime.timedelta(days=random_number_of_days)
        + datetime.timedelta(minutes=random.randrange(MINUTES_IN_HOUR * 19))
    )
    return created_at


# generate URI for events table
def generate_uri(event: str, product: str) -> str:
    if event == "product":
        return f"/{event}/{product[0]}"
    elif event == "department":
        return f"""/{event}/{product[5].lower()}/category/{product[4].lower().replace(" ", "")}/brand/{product[1].lower().replace(" ", "")}"""
    else:
        return f"/{event}"

if __name__ == "__main__":
    print(PRODUCT_LIST)