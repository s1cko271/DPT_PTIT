#!/usr/bin/env python3
"""
IMPORT SONGS WITH FEATURES
---------------------------
Script để import các bài hát và trích xuất đặc trưng âm thanh vào cơ sở dữ liệu
"""

import os
import sqlite3
import argparse
import io
import joblib
from tqdm import tqdm
import mutagen
from modified_feature_extraction import extract_features

def create_database_structure(db_path):
    """
    Tạo cấu trúc cơ sở dữ liệu nếu chưa tồn tại
    
    Parameters:
    -----------
    db_path : str
        Đường dẫn đến file cơ sở dữ liệu SQLite
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Tạo bảng songs nếu chưa tồn tại
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS songs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        title TEXT,
        artist TEXT,
        album TEXT,
        genre TEXT,
        features BLOB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Tạo index để tìm kiếm nhanh hơn
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_songs_filename ON songs (filename)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_songs_genre ON songs (genre)')
    
    conn.commit()
    conn.close()
    print(f"Cấu trúc cơ sở dữ liệu đã được tạo tại {db_path}")

def get_genre_from_path(file_path):
    """
    Lấy thể loại từ đường dẫn file (dựa vào cấu trúc thư mục)
    
    Parameters:
    -----------
    file_path : str
        Đường dẫn đến file âm thanh
        
    Returns:
    --------
    genre : str
        Thể loại nhạc được xác định từ đường dẫn
    """
    # Giả sử cấu trúc thư mục là: audio_files/genre/song.mp3
    parts = os.path.normpath(file_path).split(os.sep)
    if len(parts) >= 2:
        return parts[-2]  # Lấy tên thư mục cha
    return "Unknown"

def import_songs(audio_dir, db_path, force_update=False):
    """
    Import các bài hát từ thư mục vào cơ sở dữ liệu và trích xuất đặc trưng
    
    Parameters:
    -----------
    audio_dir : str
        Đường dẫn đến thư mục chứa các file âm thanh
    db_path : str
        Đường dẫn đến file cơ sở dữ liệu SQLite
    force_update : bool
        Cập nhật lại đặc trưng cho các bài hát đã tồn tại
    """
    # Tạo cấu trúc database nếu chưa có
    create_database_structure(db_path)
    
    # Kết nối đến cơ sở dữ liệu
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Lấy danh sách các file âm thanh
    audio_files = []
    
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(('.mp3', '.wav', '.ogg', '.flac')):
                audio_files.append(os.path.join(root, file))
    
    print(f"Tìm thấy {len(audio_files)} file âm thanh")
    
    # Kiểm tra xem file đã tồn tại trong database chưa
    if not force_update:
        existing_files = {}
        cursor.execute("SELECT filename FROM songs")
        for row in cursor.fetchall():
            existing_files[row[0]] = True
        
        # Lọc ra những file chưa được import
        new_audio_files = []
        for file_path in audio_files:
            filename = os.path.basename(file_path)
            if filename not in existing_files:
                new_audio_files.append(file_path)
        
        print(f"Có {len(new_audio_files)} file mới cần import")
        audio_files = new_audio_files
    
    # Trích xuất đặc trưng và lưu vào cơ sở dữ liệu
    success_count = 0
    error_count = 0
    
    for audio_path in tqdm(audio_files, desc="Đang trích xuất đặc trưng"):
        try:
            # Trích xuất đặc trưng
            features = extract_features(audio_path)
            
            if features is not None:
                # Chuyển đổi đặc trưng thành dạng binary để lưu vào SQLite
                buffer = io.BytesIO()
                joblib.dump(features, buffer)
                features_blob = buffer.getvalue()
                
                # Lấy thông tin cơ bản về file
                filename = os.path.basename(audio_path)
                genre = get_genre_from_path(audio_path)
                
                # Trích xuất metadata từ file nhạc
                try:
                    audio_metadata = mutagen.File(audio_path, easy=True)
                    title = audio_metadata.get('title', [os.path.splitext(filename)[0]])[0]
                    artist = audio_metadata.get('artist', ['Unknown'])[0]
                    album = audio_metadata.get('album', ['Unknown'])[0]
                except:
                    # Nếu không đọc được metadata, sử dụng thông tin mặc định
                    title = os.path.splitext(filename)[0]
                    artist = 'Unknown'
                    album = 'Unknown'
                
                # Kiểm tra xem file đã tồn tại trong cơ sở dữ liệu chưa
                cursor.execute("SELECT id FROM songs WHERE filename = ?", (filename,))
                existing = cursor.fetchone()
                
                if existing and force_update:
                    # Cập nhật nếu đã tồn tại và yêu cầu cập nhật
                    cursor.execute(
                        "UPDATE songs SET features = ?, title = ?, artist = ?, album = ?, genre = ? WHERE filename = ?",
                        (features_blob, title, artist, album, genre, filename)
                    )
                elif not existing:
                    # Thêm mới nếu chưa tồn tại
                    cursor.execute(
                        "INSERT INTO songs (filename, title, artist, album, genre, features) VALUES (?, ?, ?, ?, ?, ?)",
                        (filename, title, artist, album, genre, features_blob)
                    )
                
                conn.commit()
                success_count += 1
            else:
                error_count += 1
        except Exception as e:
            print(f"Lỗi khi xử lý file {audio_path}: {e}")
            error_count += 1
    
    conn.close()
    
    print(f"Hoàn thành import: {success_count} thành công, {error_count} lỗi")
    return success_count, error_count

def main():
    parser = argparse.ArgumentParser(description='Import bài hát và trích xuất đặc trưng vào database')
    parser.add_argument('--dir', '-d', required=True, help='Thư mục chứa file âm thanh')
    parser.add_argument('--db', default='./database/music_features.db', help='Đường dẫn đến file cơ sở dữ liệu (mặc định: ./database/music_features.db)')
    parser.add_argument('--force', '-f', action='store_true', help='Cập nhật lại đặc trưng cho các bài hát đã tồn tại')
    
    args = parser.parse_args()
    
    # Tạo thư mục chứa db nếu chưa tồn tại
    os.makedirs(os.path.dirname(args.db), exist_ok=True)
    
    import_songs(args.dir, args.db, args.force)

if __name__ == "__main__":
    main() 