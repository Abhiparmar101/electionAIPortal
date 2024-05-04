import sqlite3

def init_db():
    with sqlite3.connect('metadata.db') as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                  cameraid TEXT,
                  timestamp TEXT,
                  imageurl TEXT,
                
                  personcount INTEGER
                  
            )
        ''')
        conn.commit()
        

