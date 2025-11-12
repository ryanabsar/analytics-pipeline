import pandas as pd
import numpy as np
from faker import Faker
import re
import argparse
import os
import 


faker = Faker()

population_df = pd.read_csv(os.path.join(file_path, "population_data.csv"))
world_pop_df = pd.read_csv(os.path.join(file_path, "world_pop.csv"))
distribution_centers_df = pd.read_csv(os.path.join(file_path, "distribution_centers.csv"))

def sanitize(arr: np.ndarray) -> np.ndarray:
    return np.vectorize(lambda x: re.sub(r"[^a-zA-Z]", "", x).lower())(arr)

def generate_addresses(n_users:int, LOC_DATA:pd.DataFrame = world_pop_df) -> pd.DataFrame:
    weights = LOC_DATA['population'] / LOC_DATA['population'].sum()
    # sample n_users based on population weights
    df_sampled = LOC_DATA.sample(n=n_users, weights=weights, replace=True).reset_index(drop=True)
    # generate street addresses
    df_sampled['street'] = [faker.street_address() for _ in range(n_users)]

    # return n_users sampled addresses based on population weights
    return df_sampled



def users(n_users:int, list_of_names: np.ndarray, list_of_address: pd.DataFrame) -> pd.DataFrame:
    faker_params = np.array([[faker.first_name(), faker.last_name_nonbinary(), faker.uuid4()] for _ in range(n_users)])

    emails = np.char.lower(
        sanitize(faker_params[:,0]) + "." + sanitize(faker_params[:,1]) + "@example.com"
    )

    df = pd.DataFrame({
        "id": np.arange(1, n_users + 1),
        "first_name": faker_params[:,0],
        "last_name": faker_params[:,1],
        "email": emails,
        "age": np.random.randint(18, 70, size=n_users),
        "gender": np.random.choice(["M", "F", "Non-binary"], size=n_users),
        "state": list_of_address['state'],
        "street_address": list_of_address['street'],
        "postal_code": list_of_address['postal_code'],
        "city": list_of_address['city'],
        "country": list_of_address['country'],
        "latitude": list_of_address['latitude'],
        "longitude": list_of_address['longitude'],
        "num_of_orders": np.random.choice([0,1,2,3,4,5], p=[0.2, 0.5, 0.2, 0.05, 0.05], size=n_users),
        "created_at": pd.to_datetime("2019-01-01") + pd.to_timedelta((np.random.rand(n_users) ** 2) * 365 * 5, unit='d'), # Bias towards older dates
    })
    return df

def order(users_df: pd.DataFrame) -> pd.DataFrame:
    orders_df = users_df.loc[users_df.index.repeat(users_df['num_of_orders'])].copy()

    #add prefix users_ to avoid confusion
    orders_df = orders_df.add_prefix("user_")
    orders_df["id"] = np.arange(1, len(orders_df) + 1)
    # order created at is after user created at
    time_between_dates = (pd.to_datetime("today") - orders_df["user_created_at"]).dt.days
    orders_df["created_at"] = orders_df["user_created_at"] + pd.to_timedelta(np.random.randint(1, time_between_dates + 1), unit="D")
    orders_df["status"] = np.random.choice(["Complete", "Cancelled", "Returned", "Processing", "Shipped"],
                                           p=[0.25, 0.15, 0.1, 0.2, 0.3], size=len(orders_df))
    # add random generator for days it takes to ship deliver, return etc.

    # initialize shipped_at, delivered_at, returned_at columns
    orders_df["shipped_at"] = pd.NaT
    orders_df["delivered_at"] = pd.NaT
    orders_df["returned_at"] = pd.NaT

    # vectorized masked for each status
    mask_shipped = orders_df['status'] == "Shipped"
    mask_complete = orders_df['status'] == "Complete"
    mask_returned = orders_df['status'] == "Returned"

    # compute dates for each subset
    for mask, delivered, returned in [
        (mask_returned, True, True),
        (mask_complete, True, False),
        (mask_shipped, False, False)
    ]:
        size = mask.sum()
        if size > 0:
            orders_df.loc[mask, "shipped_at"] = orders_df.loc[mask, "created_at"] + pd.to_timedelta(np.random.randint(0, 3 * 24), unit="h") # 0-3 days
            if delivered:
                orders_df.loc[mask, "delivered_at"] = orders_df.loc[mask, "shipped_at"] + pd.to_timedelta(np.random.randint(0, 5 * 24), unit="h") # 0-5 days
            if returned:
                orders_df.loc[mask, "returned_at"] = orders_df.loc[mask, "delivered_at"] + pd.to_timedelta(np.random.randint(0, 3 * 24), unit="h") # 0-3 days

    # number of items per order given a weighted distribution
    orders_df["num_of_items"] = np.random.choice([1,2,3,4], p=[0.7, 0.2, 0.05, 0.05], size=len(orders_df))

    # delete column with prefix user_
    orders_df = orders_df.drop(columns=[col for col in orders_df.columns if col.startswith("user_")])

    return orders_df

def order_items(orders_df: pd.DataFrame) -> pd.DataFrame:
    # repeat orders based on num_of_items from orders_df
    order_items_df = orders_df.loc[orders_df.index.repeat(orders_df['num_of_items'])].copy()
    order_items_df = order_items_df.add_prefix("order_") # add prefix order_ to avoid confusion

    order_items_df["id"] = np.arange(1, len(order_items_df) + 1)

def events(order_items_df: pd.DataFrame) -> pd.DataFrame:
    pass

def inventory_items() -> pd.DataFrame:
    pass



def main(n_users: int):
    
    list_of_address = generate_addresses(n_users)
    users_df = users(n_users, list_of_names, list_of_address)
    orders_df = order(users_df)

if __name__ == "__main__":
    main(

    )