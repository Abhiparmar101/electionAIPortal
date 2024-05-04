import pandas as pd
import sqlite3

# Define your column names
column_names = ['SR', 'AC_Name', 'Part_No_Polling_Station_Name', 'Technician_Contact_Number', 'Technican_Name', 'cameraid', 'zone']

# Path to your CSV file
csv_file_path = 'camera_details.csv'

# Read the CSV file, assuming it doesn't contain headers
data = pd.read_csv(csv_file_path, names=column_names, header=None)

# Connect to your SQLite database
db_path = 'metadata.db'
conn = sqlite3.connect(db_path)

# Create a cursor object using the cursor method
cursor = conn.cursor()

# Create table SQL statement
create_table_sql = f"""
CREATE TABLE IF NOT EXISTS camera_installation_details (
    SR TEXT,
    AC_Name TEXT,
    Part_No_Polling_Station_Name TEXT,
    Technician_Contact_Number TEXT,
    Technican_Name TEXT,
    cameraid TEXT,
    zone TEXT
)
"""
cursor.execute(create_table_sql)

# Prepare the INSERT INTO statement
insert_sql = f"""
INSERT INTO camera_installation_details (SR, AC_Name, Part_No_Polling_Station_Name, Technician_Contact_Number, Technican_Name, cameraid, zone)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""

# Insert data from DataFrame to the table
for _, row in data.iterrows():
    cursor.execute(insert_sql, row.values)

# Commit the changes
conn.commit()

# Verify the insertion by querying the newly created table
query_result = pd.read_sql_query("SELECT * FROM camera_installation_details", conn)
print("Database table contents:")
print(query_result.head())

# Close the connection
conn.close()

