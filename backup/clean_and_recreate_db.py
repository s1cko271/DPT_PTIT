#!/usr/bin/env python3
"""
CLEAN AND RECREATE DATABASE
---------------------------
Script để làm sạch và tái tạo cơ sở dữ liệu, đồng thời tổ chức lại file nhạc theo thể loại
"""

import os
import sqlite3
import argparse
import shutil
from tqdm import tqdm

def clean_database(db_path):
    """
    Xóa và tạo lại database trắng
    
    Parameters:
    -----------
    db_path : str
        Đường dẫn đến file cơ sở dữ liệu SQLite
    """
    # Xóa file cũ nếu tồn tại
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Đã xóa file database cũ: {db_path}")
    
    # Tạo database mới
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Tạo bảng songs
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
    
    print(f"Đã tạo database mới tại: {db_path}")

def organize_files_by_genre(audio_dir, clean=False):
    """
    Tổ chức lại các file âm thanh theo thể loại
    
    Parameters:
    -----------
    audio_dir : str
        Đường dẫn đến thư mục gốc chứa các file âm thanh
    clean : bool
        Nếu True, sẽ xóa tất cả các file hiện có trước khi tổ chức lại
    """
    import mutagen
    from mutagen.easyid3 import EasyID3
    from mutagen.id3 import ID3NoHeaderError
    
    # Tạo thư mục gốc nếu chưa tồn tại
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
        print(f"Đã tạo thư mục: {audio_dir}")
    
    # Xóa tất cả các file hiện có nếu yêu cầu
    if clean:
        for item in os.listdir(audio_dir):
            item_path = os.path.join(audio_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
        print(f"Đã xóa tất cả các file trong thư mục: {audio_dir}")
    
    # Lấy danh sách tất cả các file âm thanh
    audio_files = []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(('.mp3', '.wav', '.ogg', '.flac')):
                audio_files.append(os.path.join(root, file))
    
    print(f"Tìm thấy {len(audio_files)} file âm thanh")
    
    # Tổ chức lại theo thể loại
    for audio_path in tqdm(audio_files, desc="Đang tổ chức file"):
        # Kiểm tra xem file đã được tổ chức trong thư mục thể loại chưa
        current_dir = os.path.dirname(audio_path)
        if current_dir != audio_dir and os.path.dirname(current_dir) == audio_dir:
            # File đã nằm trong thư mục thể loại, bỏ qua
            continue
        
        # Trích xuất thể loại từ metadata
        try:
            audio = mutagen.File(audio_path, easy=True)
            genre = audio.get('genre', ['Unknown'])[0]
        except:
            genre = "Unknown"
        
        # Chuẩn hóa tên thể loại
        genre = genre.strip().lower()
        if not genre or genre == "":
            genre = "Unknown"
        
        # Map các thể loại tương tự
        genre_mapping = {
            'classical': 'classical',
            'classic': 'classical',
            'blues': 'blues',
            'country': 'country',
            'disco': 'disco',
            'hiphop': 'hiphop',
            'hip hop': 'hiphop',
            'jazz': 'jazz',
            'metal': 'metal',
            'pop': 'pop',
            'reggae': 'reggae',
            'rock': 'rock'
        }
        
        # Áp dụng mapping
        for key, value in genre_mapping.items():
            if key in genre:
                genre = value
                break
        
        # Tạo thư mục thể loại nếu chưa tồn tại
        genre_dir = os.path.join(audio_dir, genre)
        if not os.path.exists(genre_dir):
            os.makedirs(genre_dir)
        
        # Di chuyển file vào thư mục thể loại
        new_path = os.path.join(genre_dir, os.path.basename(audio_path))
        
        # Đảm bảo không ghi đè file
        counter = 1
        base_name, ext = os.path.splitext(new_path)
        while os.path.exists(new_path) and new_path != audio_path:
            new_path = f"{base_name}_{counter}{ext}"
            counter += 1
        
        # Di chuyển file
        if new_path != audio_path:
            shutil.move(audio_path, new_path)
    
    # Đếm số lượng file theo thể loại
    genre_counts = {}
    for item in os.listdir(audio_dir):
        item_path = os.path.join(audio_dir, item)
        if os.path.isdir(item_path):
            count = len([f for f in os.listdir(item_path) 
                        if f.endswith(('.mp3', '.wav', '.ogg', '.flac'))])
            genre_counts[item] = count
    
    print("\nSố lượng file theo thể loại:")
    for genre, count in sorted(genre_counts.items()):
        print(f"  - {genre}: {count} file")
    
    return genre_counts

def main():
    parser = argparse.ArgumentParser(description='Làm sạch và tái tạo cơ sở dữ liệu, tổ chức lại file nhạc theo thể loại')
    parser.add_argument('--dir', '-d', default='./audio_files', help='Thư mục chứa file âm thanh (mặc định: ./audio_files)')
    parser.add_argument('--db', default='./database/music_features.db', help='Đường dẫn đến file cơ sở dữ liệu (mặc định: ./database/music_features.db)')
    parser.add_argument('--clean', '-c', action='store_true', help='Xóa tất cả các file hiện có trước khi tổ chức lại')
    
    args = parser.parse_args()
    
    # Tạo thư mục chứa db nếu chưa tồn tại
    os.makedirs(os.path.dirname(args.db), exist_ok=True)
    
    # Làm sạch và tạo lại database
    clean_database(args.db)
    
    # Tổ chức lại file theo thể loại
    organize_files_by_genre(args.dir, args.clean)
    
    print("\nHoàn thành quá trình làm sạch và tái tạo")

if __name__ == "__main__":
    main() 