import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Loading environment variables from .env file
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)

db = client["insurance_fraud"]

policies_col = db["policies"]
policyholders_col = db["policyholders"]
vehicles_col = db["vehicles"]
claims_col = db["claims"]
events_col = db["claim_events"]