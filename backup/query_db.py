#!/usr/bin/env python3
"""
QUERY DATABASE
--------------
Script để tìm kiếm trong cơ sở dữ liệu nhạc
"""

import os
import sqlite3
import argparse
import io
import joblib
import numpy as np
from tabulate import tabulate

def list_genres(db_path):
    """
    Liệt kê các thể loại trong cơ sở dữ liệu
    
    Parameters:
    -----------
    db_path : str
        Đường dẫn đến file cơ sở dữ liệu SQLite
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT DISTINCT genre FROM songs")
    genres = cursor.fetchall()
    
    conn.close()
    
    if not genres:
        print("Không tìm thấy thể loại nào trong cơ sở dữ liệu")
        return []
    
    genres = [g[0] for g in genres if g[0] is not None]
    genres.sort()
    
    return genres

def search_by_genre(db_path, genre, limit=10):
    """
    Tìm kiếm bài hát theo thể loại
    
    Parameters:
    -----------
    db_path : str
        Đường dẫn đến file cơ sở dữ liệu SQLite
    genre : str
        Thể loại cần tìm kiếm
    limit : int
        Số lượng kết quả tối đa
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT id, filename, title, artist, album 
    FROM songs 
    WHERE genre LIKE ? 
    ORDER BY title
    LIMIT ?
    """, (f"%{genre}%", limit))
    
    results = cursor.fetchall()
    conn.close()
    
    return results

def search_by_title(db_path, title_query, limit=10):
    """
    Tìm kiếm bài hát theo tiêu đề
    
    Parameters:
    -----------
    db_path : str
        Đường dẫn đến file cơ sở dữ liệu SQLite
    title_query : str
        Từ khóa tiêu đề cần tìm kiếm
    limit : int
        Số lượng kết quả tối đa
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT id, filename, title, artist, album, genre 
    FROM songs 
    WHERE title LIKE ? 
    ORDER BY title
    LIMIT ?
    """, (f"%{title_query}%", limit))
    
    results = cursor.fetchall()
    conn.close()
    
    return results

def search_by_artist(db_path, artist_query, limit=10):
    """
    Tìm kiếm bài hát theo nghệ sĩ
    
    Parameters:
    -----------
    db_path : str
        Đường dẫn đến file cơ sở dữ liệu SQLite
    artist_query : str
        Từ khóa nghệ sĩ cần tìm kiếm
    limit : int
        Số lượng kết quả tối đa
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT id, filename, title, artist, album, genre 
    FROM songs 
    WHERE artist LIKE ? 
    ORDER BY artist, title
    LIMIT ?
    """, (f"%{artist_query}%", limit))
    
    results = cursor.fetchall()
    conn.close()
    
    return results

def get_song_details(db_path, song_id):
    """
    Lấy thông tin chi tiết về một bài hát
    
    Parameters:
    -----------
    db_path : str
        Đường dẫn đến file cơ sở dữ liệu SQLite
    song_id : int
        ID của bài hát cần lấy thông tin
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT id, filename, title, artist, album, genre, features 
    FROM songs 
    WHERE id = ?
    """, (song_id,))
    
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        return None
    
    # Giải nén dữ liệu đặc trưng
    features = joblib.load(io.BytesIO(result[6]))
    
    return {
        'id': result[0],
        'filename': result[1],
        'title': result[2],
        'artist': result[3],
        'album': result[4],
        'genre': result[5],
        'features': features
    }

def search_similar_songs(db_path, song_id, metric='cosine', limit=5):
    """
    Tìm kiếm các bài hát tương tự dựa trên đặc trưng âm thanh
    
    Parameters:
    -----------
    db_path : str
        Đường dẫn đến file cơ sở dữ liệu SQLite
    song_id : int
        ID của bài hát gốc
    metric : str
        Phương pháp đo lường ('cosine' hoặc 'euclidean')
    limit : int
        Số lượng kết quả tối đa
    """
    # Lấy thông tin bài hát gốc
    source_song = get_song_details(db_path, song_id)
    if not source_song:
        print(f"Không tìm thấy bài hát có ID = {song_id}")
        return []
    
    # Lấy vector đặc trưng của bài hát gốc
    source_vector = source_song['features']['feature_vector']
    
    # Lấy tất cả các bài hát từ database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, filename, title, artist, genre, features FROM songs")
    songs = cursor.fetchall()
    
    conn.close()
    
    # Tính toán độ tương đồng
    similarities = []
    
    for song in songs:
        if song[0] == song_id:  # Bỏ qua bài hát gốc
            continue
        
        # Giải nén đặc trưng
        song_features = joblib.load(io.BytesIO(song[5]))
        song_vector = song_features['feature_vector']
        
        # Chuẩn hóa các vector
        source_norm = source_vector / np.linalg.norm(source_vector)
        song_norm = song_vector / np.linalg.norm(song_vector)
        
        if metric == 'cosine':
            # Tính độ tương đồng cosine
            sim = np.dot(source_norm, song_norm)
            # Sắp xếp giảm dần (càng cao càng giống)
            similarities.append((song, sim, True))
        else:
            # Tính khoảng cách Euclidean
            dist = np.linalg.norm(source_vector - song_vector)
            # Sắp xếp tăng dần (càng thấp càng giống)
            similarities.append((song, dist, False))
    
    # Sắp xếp kết quả
    if metric == 'cosine':
        similarities.sort(key=lambda x: x[1], reverse=True)
    else:
        similarities.sort(key=lambda x: x[1])
    
    # Giới hạn số lượng kết quả
    results = similarities[:limit]
    
    return results

def print_song_table(songs, with_id=True, with_genre=True):
    """
    In bảng thông tin các bài hát
    
    Parameters:
    -----------
    songs : list
        Danh sách các bài hát
    with_id : bool
        Hiển thị cột ID
    with_genre : bool
        Hiển thị cột thể loại
    """
    if not songs:
        print("Không có kết quả nào được tìm thấy.")
        return
    
    headers = []
    rows = []
    
    if with_id:
        headers.append("ID")
    
    headers.extend(["Title", "Artist", "Album"])
    
    if with_genre:
        headers.append("Genre")
    
    for song in songs:
        row = []
        if with_id:
            row.append(song[0])
        
        row.append(song[2] if song[2] else "Unknown")  # Title
        row.append(song[3] if song[3] else "Unknown")  # Artist
        row.append(song[4] if song[4] else "Unknown")  # Album
        
        if with_genre and len(song) > 5:
            row.append(song[5] if song[5] else "Unknown")  # Genre
        
        rows.append(row)
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))

def print_similar_songs(similar_songs, metric):
    """
    In bảng các bài hát tương tự
    
    Parameters:
    -----------
    similar_songs : list
        Danh sách các bài hát tương tự
    metric : str
        Phương pháp đo lường ('cosine' hoặc 'euclidean')
    """
    if not similar_songs:
        print("Không tìm thấy bài hát tương tự.")
        return
    
    headers = ["ID", "Title", "Artist", "Genre"]
    if metric == 'cosine':
        headers.append("Similarity")
    else:
        headers.append("Distance")
    
    rows = []
    
    for song, value, is_similarity in similar_songs:
        row = [
            song[0],           # ID
            song[2] or "Unknown", # Title
            song[3] or "Unknown", # Artist
            song[4] or "Unknown"  # Genre
        ]
        
        # Định dạng giá trị tương đồng
        if is_similarity:  # Cosine
            row.append(f"{value:.4f}")
        else:  # Euclidean
            row.append(f"{value:.4f}")
        
        rows.append(row)
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))

def main():
    parser = argparse.ArgumentParser(description='Tìm kiếm trong cơ sở dữ liệu nhạc')
    parser.add_argument('--db', default='./database/music_features.db', help='Đường dẫn đến file cơ sở dữ liệu (mặc định: ./database/music_features.db)')
    
    subparsers = parser.add_subparsers(dest='command', help='Các lệnh')
    
    # Lệnh liệt kê thể loại
    genres_parser = subparsers.add_parser('genres', help='Liệt kê các thể loại')
    
    # Lệnh tìm kiếm theo thể loại
    genre_parser = subparsers.add_parser('genre', help='Tìm kiếm theo thể loại')
    genre_parser.add_argument('query', help='Thể loại cần tìm kiếm')
    genre_parser.add_argument('--limit', '-l', type=int, default=10, help='Số lượng kết quả tối đa')
    
    # Lệnh tìm kiếm theo tiêu đề
    title_parser = subparsers.add_parser('title', help='Tìm kiếm theo tiêu đề')
    title_parser.add_argument('query', help='Từ khóa tiêu đề cần tìm kiếm')
    title_parser.add_argument('--limit', '-l', type=int, default=10, help='Số lượng kết quả tối đa')
    
    # Lệnh tìm kiếm theo nghệ sĩ
    artist_parser = subparsers.add_parser('artist', help='Tìm kiếm theo nghệ sĩ')
    artist_parser.add_argument('query', help='Từ khóa nghệ sĩ cần tìm kiếm')
    artist_parser.add_argument('--limit', '-l', type=int, default=10, help='Số lượng kết quả tối đa')
    
    # Lệnh xem thông tin chi tiết bài hát
    details_parser = subparsers.add_parser('details', help='Xem thông tin chi tiết bài hát')
    details_parser.add_argument('id', type=int, help='ID của bài hát')
    
    # Lệnh tìm kiếm bài hát tương tự
    similar_parser = subparsers.add_parser('similar', help='Tìm kiếm bài hát tương tự')
    similar_parser.add_argument('id', type=int, help='ID của bài hát gốc')
    similar_parser.add_argument('--metric', '-m', choices=['cosine', 'euclidean'], default='cosine',
                              help='Phương pháp đo lường (mặc định: cosine)')
    similar_parser.add_argument('--limit', '-l', type=int, default=5, help='Số lượng kết quả tối đa')
    
    args = parser.parse_args()
    
    # Kiểm tra database tồn tại
    if not os.path.exists(args.db):
        print(f"Lỗi: Không tìm thấy file cơ sở dữ liệu tại {args.db}")
        return
    
    # Xử lý các lệnh
    if args.command == 'genres':
        genres = list_genres(args.db)
        if genres:
            print("Danh sách thể loại trong cơ sở dữ liệu:")
            for genre in genres:
                print(f"  - {genre}")
    
    elif args.command == 'genre':
        results = search_by_genre(args.db, args.query, args.limit)
        print(f"Kết quả tìm kiếm cho thể loại '{args.query}':")
        print_song_table(results, with_genre=False)
    
    elif args.command == 'title':
        results = search_by_title(args.db, args.query, args.limit)
        print(f"Kết quả tìm kiếm cho tiêu đề '{args.query}':")
        print_song_table(results)
    
    elif args.command == 'artist':
        results = search_by_artist(args.db, args.query, args.limit)
        print(f"Kết quả tìm kiếm cho nghệ sĩ '{args.query}':")
        print_song_table(results)
    
    elif args.command == 'details':
        song = get_song_details(args.db, args.id)
        if song:
            print(f"Thông tin chi tiết bài hát (ID: {song['id']}):")
            print(f"  - Tiêu đề: {song['title']}")
            print(f"  - Nghệ sĩ: {song['artist']}")
            print(f"  - Album: {song['album']}")
            print(f"  - Thể loại: {song['genre']}")
            print(f"  - Filename: {song['filename']}")
            print("\nĐặc trưng âm thanh:")
            for key, value in song['features'].items():
                if key != 'feature_vector':  # Không in vector đặc trưng
                    print(f"  - {key}: {value}")
        else:
            print(f"Không tìm thấy bài hát có ID = {args.id}")
    
    elif args.command == 'similar':
        results = search_similar_songs(args.db, args.id, args.metric, args.limit)
        if results:
            if args.metric == 'cosine':
                print(f"Các bài hát tương tự (sử dụng Cosine similarity, ID gốc: {args.id}):")
            else:
                print(f"Các bài hát tương tự (sử dụng Euclidean distance, ID gốc: {args.id}):")
            print_similar_songs(results, args.metric)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 