import sqlite3
import os
import joblib
import io
from tqdm import tqdm

def merge_databases(db1_path, db2_path, output_path):
    """
    Tích hợp hai cơ sở dữ liệu SQLite vào một cơ sở dữ liệu mới
    
    Parameters:
    -----------
    db1_path : str
        Đường dẫn đến cơ sở dữ liệu thứ nhất
    db2_path : str
        Đường dẫn đến cơ sở dữ liệu thứ hai
    output_path : str
        Đường dẫn đến cơ sở dữ liệu đầu ra
    """
    # Kết nối đến các cơ sở dữ liệu
    conn1 = sqlite3.connect(db1_path)
    conn2 = sqlite3.connect(db2_path)
    conn_out = sqlite3.connect(output_path)
    
    cursor1 = conn1.cursor()
    cursor2 = conn2.cursor()
    cursor_out = conn_out.cursor()
    
    # Tạo bảng trong cơ sở dữ liệu mới
    cursor_out.execute('''
    CREATE TABLE IF NOT EXISTS songs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        title TEXT,
        artist TEXT,
        album TEXT,
        features BLOB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Lấy tất cả bài hát từ cơ sở dữ liệu thứ nhất
    cursor1.execute("SELECT filename, title, artist, album, features FROM songs")
    songs1 = cursor1.fetchall()
    
    # Lấy tất cả bài hát từ cơ sở dữ liệu thứ hai
    cursor2.execute("SELECT filename, title, artist, album, features FROM songs")
    songs2 = cursor2.fetchall()
    
    # Tạo set để theo dõi các file đã được thêm vào
    added_files = set()
    
    # Thêm các bài hát từ cơ sở dữ liệu thứ nhất
    print("Đang thêm bài hát từ cơ sở dữ liệu thứ nhất...")
    for song in tqdm(songs1):
        filename, title, artist, album, features = song
        if filename not in added_files:
            cursor_out.execute(
                "INSERT INTO songs (filename, title, artist, album, features) VALUES (?, ?, ?, ?, ?)",
                (filename, title, artist, album, features)
            )
            added_files.add(filename)
    
    # Thêm các bài hát từ cơ sở dữ liệu thứ hai
    print("Đang thêm bài hát từ cơ sở dữ liệu thứ hai...")
    for song in tqdm(songs2):
        filename, title, artist, album, features = song
        if filename not in added_files:
            cursor_out.execute(
                "INSERT INTO songs (filename, title, artist, album, features) VALUES (?, ?, ?, ?, ?)",
                (filename, title, artist, album, features)
            )
            added_files.add(filename)
    
    # Lưu thay đổi và đóng kết nối
    conn_out.commit()
    conn1.close()
    conn2.close()
    conn_out.close()
    
    print(f"Đã tích hợp thành công {len(added_files)} bài hát vào cơ sở dữ liệu mới.")

if __name__ == "__main__":
    # Đường dẫn đến các cơ sở dữ liệu
    db1_path = os.path.join("database", "music_features.db")
    db2_path = os.path.join("database", "music_database.db")
    output_path = os.path.join("database", "merged_music.db")
    
    # Tích hợp cơ sở dữ liệu
    merge_databases(db1_path, db2_path, output_path) 