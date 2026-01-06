import pandas as pd
from pymongo import MongoClient

# 1. MONGODB CONFIGURATION
# Replace <password> with your actual database user password
MONGO_URI = "mongodb+srv://vishwavarshaa7_db_user:ZQYT2zUjdac4qJbG@cluster0.ogxvqwx.mongodb.net/?retryWrites=true&w=majority"
DB_NAME = "KabaddiDB"

# 2. FILE PATHS (Ensure these files are in the same folder as this script)
MATCH_FILE = "match point (kabbadi) final.xlsx"
ATTENDANCE_FILE = "Copy of Copy_of_Copy_of_PAX_-_Attendance_1111_processed_v2(1).xlsx"

def connect_to_db():
    client = MongoClient(MONGO_URI)
    return client[DB_NAME]

def upload_match_data(db):
    print("Reading Match Scores...")
    xls = pd.ExcelFile(MATCH_FILE)
    all_records = []

    for sheet_name in xls.sheet_names:
        # Skip summary sheets if they exist
        if "TOTAL" in sheet_name.upper(): continue
        
        df = pd.read_excel(xls, sheet_name=sheet_name)
        # Cleaning logic similar to your dashboard
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if df.empty: continue
        
        # Set first row as header if necessary (common in your files)
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        
        # Add tournament identifier
        df["Tournament"] = sheet_name
        
        # Convert to dictionary and add to list
        all_records.extend(df.to_dict(orient='records'))

    if all_records:
        db.match_scores.delete_many({}) # Clears old data before upload
        db.match_scores.insert_many(all_records)
        print(f"✅ Uploaded {len(all_records)} match records.")

def upload_attendance_data(db):
    print("Reading Attendance Data...")
    xls = pd.ExcelFile(ATTENDANCE_FILE)
    all_records = []

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if df.empty: continue
        
        # Add Month/Sheet identifier
        df["Source_Month"] = sheet_name
        
        all_records.extend(df.to_dict(orient='records'))

    if all_records:
        db.attendance.delete_many({}) # Clears old data before upload
        db.attendance.insert_many(all_records)
        print(f"✅ Uploaded {len(all_records)} attendance records.")

if __name__ == "__main__":
    try:
        database = connect_to_db()
        upload_match_data(database)
        upload_attendance_data(database)
        print("\n🚀 All data successfully migrated to MongoDB Atlas Cluster0!")
    except Exception as e:
        print(f"❌ Error: {e}")