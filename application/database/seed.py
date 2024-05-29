import os
import pandas as pd
from sqlalchemy import create_engine, Table, Column, MetaData, Integer
from dotenv import load_dotenv

load_dotenv()
DB_HOST = os.getenv("DB_HOST")

DB_ENGINE = create_engine(DB_HOST)
metadata = MetaData()
nba_table = Table("nba_stats", metadata, Column("id", Integer, primary_key=True))
metadata.create_all(DB_ENGINE)
data = pd.read_csv("path")
data.to_sql("nba_stats", DB_ENGINE, if_exists="fail", index=False)
print("Database seeding completed.")
