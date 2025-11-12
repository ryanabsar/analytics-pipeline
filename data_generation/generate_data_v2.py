import pandas as pd
import numpy as np
from faker import Faker
import re
import os

class DataGenerator:
    def __init__(self, n_users:int, seed:int = 42):
        self.n_users = n_users
        self.faker = Faker()
        Faker.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def sanitize(arr: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda x: re.sub(r"[^a-zA-Z]", "", x).lower())(arr)
    
    @staticmethod
    def add_column_prefix(df: pd.DataFrame, prefix: str, exclude: list[str] = []) -> pd.DataFrame:
        df.columns = [
            f"{prefix}_{col}" if col not in exclude else col
            for col in df.columns
        ]
        return df

    
    def generate_addresses(self, LOC_DATA:pd.DataFrame) -> pd.DataFrame:
        weights = LOC_DATA['population'] / LOC_DATA['population'].sum()
        # sample n_users based on population weights
        df_sampled = LOC_DATA.sample(n=self.n_users, weights=weights, replace=True).reset_index(drop=True)
        # generate street addresses
        df_sampled['street'] = [self.faker.street_address() for _ in range(self.n_users)]

        # return n_users sampled addresses based on population weights
        return df_sampled
    
    def generate_stock(self, products_df: pd.DataFrame) -> pd.DataFrame:
        """Generate a stock pool for each product with random quantities."""
        stock_levels = np.random.randint(10, 50, size=len(products_df))
        stock_df = products_df.loc[products_df.index.repeat(stock_levels)].copy()

        stock_df["inventory_item_id"] = np.arange(1, len(stock_df) + 1)
        stock_df["is_sold"] = False
        stock_df["sold_at"] = pd.NaT
        stock_df["related_order_item_id"] = pd.NA
        stock_df["related_order_id"] = pd.NA
        stock_df["user_id"] = pd.NA
        stock_df["created_at"] = pd.to_datetime("2019-01-01") + pd.to_timedelta(
            np.random.randint(0, 365 * 5, size=len(stock_df)), unit="D"
        )
        self.stock_df = stock_df
        return stock_df

    def generate_users(self, addresses: pd.DataFrame) -> pd.DataFrame:
        gender_list = np.random.choice(["M", "F"], size=self.n_users)

        # generate names based on gender vectorized
        first_names = np.where(gender_list == "M",
                               np.array([self.faker.first_name_male() for _ in range(self.n_users)]),
                               np.array([self.faker.first_name_female() for _ in range(self.n_users)]))
        last_names = [self.faker.last_name_nonbinary() for _ in range(self.n_users)]

        uuids = [self.faker.uuid4() for _ in range(self.n_users)]

        emails = np.char.lower(self.sanitize(
            np.array(first_names)) + "." + self.sanitize(np.array(last_names)) + "@example.com"
            )
        
        users_df = pd.DataFrame({
            "id": np.arange(1, self.n_users + 1),
            "uuid": uuids,
            "first_name": first_names,
            "last_name": last_names,
            "email": emails,
            "age": np.random.randint(18, 70, size=self.n_users),
            "gender": gender_list,
            **addresses.to_dict(orient='list'),
            "num_of_orders": np.random.choice([0,1,2,3,4], p=[0.2, 0.5, 0.2, 0.05, 0.05], size=self.n_users),
            "created_at": pd.to_datetime("2019-01-01") + pd.to_timedelta((np.random.rand(self.n_users) ** 2) * 365 * 5, unit='d'), # Bias towards older dates
        })
        self.users_df = users_df
        return users_df
    
    def generate_orders(self) -> pd.DataFrame:
        orders_df = self.users_df.loc[self.users_df.index.repeat(self.users_df['num_of_orders'])].copy()

        #add prefix users_ to avoid confusion with other dataframes
        orders_df = orders_df.add_prefix("users_")
        orders_df.reset_index(drop=True, inplace=True)

        # generate order specific data
        orders_df['id'] = np.arange(1, len(orders_df) + 1)

        time_between_dates = (pd.to_datetime("today") - orders_df["users_created_at"]).dt.days
        orders_df["created_at"] = orders_df["users_created_at"] + pd.to_timedelta(np.random.randint(1, time_between_dates + 1), unit="D")
        orders_df["status"] = np.random.choice(["Complete", "Cancelled", "Returned", "Processing", "Shipped"],
                                           p=[0.25, 0.15, 0.1, 0.2, 0.3], size=len(orders_df))
        
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

        # delete column with prefix user_ except user_id column
        orders_df = orders_df.drop(columns=[col for col in orders_df.columns if col.startswith("user_") and col != "user_id"])

        self.orders_df = orders_df

        return orders_df
    
    def generate_order_items(self) -> pd.DataFrame:
        order_items_df = self.orders_df.loc[self.orders_df.index.repeat(self.orders_df['num_of_items'])].copy()

        #add prefix to order_ to avoid confusion except user_id
        order_items_df = self.add_column_prefix(order_items_df, prefix="order", exclude=["user_id"]) # FK user_id

        order_items_df["id"] = np.arange(1, len(order_items_df) + 1)


if __name__ == "__main__":
    # file path in ./data_generation/data
    file_path = os.path.dirname(os.path.abspath(__file__))

    products_df = pd.read_csv(os.path.join(file_path, "data", "products.csv"))
    world_pop_df = pd.read_csv(os.path.join(file_path, "data", "world_pop.csv"))
    distribution_centers_df = pd.read_csv(os.path.join(file_path, "data", "distribution_centers.csv"))

    n_users = 2
    data_generator = DataGenerator(n_users=n_users, seed=42)
    addresses = data_generator.generate_addresses(LOC_DATA=world_pop_df)
    users_df = data_generator.generate_users(addresses=addresses)
    orders_df = data_generator.generate_orders()

    # to csv
    users_df.to_csv(os.path.join(file_path, "data", "generated_users.csv"), index=False)
    orders_df.to_csv(os.path.join(file_path, "data", "generated_orders.csv"), index=False)