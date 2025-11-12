import os
from dotenv import load_dotenv

load_dotenv()

print("Loading environment variables from .env file")

def main(date):
    print(f"Main function executed with date: {date}")

if __name__ == "__main__":
    min_date = os.getenv("MIN_ORDER_DATE")
    main(min_date)

    print(min_date)