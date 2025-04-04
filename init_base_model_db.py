import os
import sqlite3
from datetime import datetime

# Create database directory
os.makedirs("data", exist_ok=True)
db_path = "data/interactions.db"

# Initialize database and tables
def init_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create model_versions table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_versions (
            version TEXT PRIMARY KEY,
            path TEXT,
            accuracy REAL,
            training_data_size INTEGER,
            training_date TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Check if base model entry exists
    cursor.execute("SELECT COUNT(*) FROM model_versions WHERE version = '1.0.0'")
    if cursor.fetchone()[0] == 0:
        # Add entry for our base model
        cursor.execute('''
            INSERT INTO model_versions 
            (version, path, accuracy, training_data_size, training_date)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            '1.0.0',  # version
            'models/model_1.0.0.mlmodel',  # path
            0.92,  # accuracy
            1000,  # training_data_size
            datetime.now().isoformat()  # training_date
        ))
        print("Added base model to database")
    else:
        print("Base model already exists in database")
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
