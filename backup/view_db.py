#!/usr/bin/env python3
"""
VIEW DATABASE
------------
Script để xem và hiển thị thông tin trong cơ sở dữ liệu nhạc
"""

import os
import sqlite3
import argparse
import io
import joblib
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np

def get_database_stats(db_path):
    """
    Lấy thống kê về cơ sở dữ liệu
    
    Parameters:
    -----------
    db_path : str
        Đường dẫn đến file cơ sở dữ liệu SQLite
        
    Returns:
    --------
    stats : dict
        Dictionary chứa các thông tin thống kê
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Lấy tổng số bài hát
    cursor.execute("SELECT COUNT(*) FROM songs")
    total_songs = cursor.fetchone()[0]
    
    # Lấy thống kê theo thể loại
    cursor.execute("""
    SELECT genre, COUNT(*) as count 
    FROM songs 
    WHERE genre IS NOT NULL 
    GROUP BY genre 
    ORDER BY count DESC
    """)
    genre_stats = cursor.fetchall()
    
    # Lấy thống kê theo nghệ sĩ (top 10)
    cursor.execute("""
    SELECT artist, COUNT(*) as count 
    FROM songs 
    WHERE artist IS NOT NULL AND artist != 'Unknown' 
    GROUP BY artist 
    ORDER BY count DESC
    LIMIT 10
    """)
    artist_stats = cursor.fetchall()
    
    conn.close()
    
    return {
        'total_songs': total_songs,
        'genre_stats': genre_stats,
        'artist_stats': artist_stats
    }

def list_all_songs(db_path, limit=None, offset=0, sort_by='id', genre=None):
    """
    Liệt kê tất cả các bài hát trong cơ sở dữ liệu
    
    Parameters:
    -----------
    db_path : str
        Đường dẫn đến file cơ sở dữ liệu SQLite
    limit : int, optional
        Số lượng bài hát tối đa để hiển thị
    offset : int
        Vị trí bắt đầu
    sort_by : str
        Trường để sắp xếp ('id', 'title', 'artist', 'genre')
    genre : str, optional
        Lọc theo thể loại
        
    Returns:
    --------
    songs : list
        Danh sách các bài hát
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Xác định câu truy vấn SQL
    query = "SELECT id, filename, title, artist, album, genre FROM songs"
    params = []
    
    # Thêm điều kiện lọc theo thể loại nếu có
    if genre:
        query += " WHERE genre LIKE ?"
        params.append(f"%{genre}%")
    
    # Xác định cách sắp xếp
    if sort_by == 'title':
        query += " ORDER BY title"
    elif sort_by == 'artist':
        query += " ORDER BY artist"
    elif sort_by == 'genre':
        query += " ORDER BY genre"
    else:  # Mặc định sắp xếp theo ID
        query += " ORDER BY id"
    
    # Thêm giới hạn và offset
    query += " LIMIT ? OFFSET ?"
    
    # Nếu không có limit, mặc định hiển thị tất cả
    if limit is None:
        limit = -1
    
    params.extend([limit, offset])
    
    # Thực thi truy vấn
    cursor.execute(query, params)
    songs = cursor.fetchall()
    
    conn.close()
    
    return songs

def print_song_table(songs):
    """
    In bảng thông tin các bài hát
    
    Parameters:
    -----------
    songs : list
        Danh sách các bài hát
    """
    if not songs:
        print("Không có bài hát nào trong cơ sở dữ liệu.")
        return
    
    headers = ["ID", "Title", "Artist", "Album", "Genre"]
    
    rows = []
    for song in songs:
        row = [
            song[0],                              # ID
            song[2] if song[2] else "Unknown",    # Title
            song[3] if song[3] else "Unknown",    # Artist
            song[4] if song[4] else "Unknown",    # Album
            song[5] if song[5] else "Unknown"     # Genre
        ]
        rows.append(row)
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))

def visualize_genre_distribution(db_path, output_file=None):
    """
    Trực quan hóa phân bố thể loại trong cơ sở dữ liệu
    
    Parameters:
    -----------
    db_path : str
        Đường dẫn đến file cơ sở dữ liệu SQLite
    output_file : str, optional
        Đường dẫn để lưu file hình ảnh
    """
    stats = get_database_stats(db_path)
    genre_stats = stats['genre_stats']
    
    if not genre_stats:
        print("Không có dữ liệu thể loại để hiển thị.")
        return
    
    # Chuẩn bị dữ liệu cho biểu đồ
    genres = [g[0] for g in genre_stats]
    counts = [g[1] for g in genre_stats]
    
    # Tạo biểu đồ
    plt.figure(figsize=(10, 6))
    plt.bar(genres, counts, color='skyblue')
    plt.xlabel('Thể loại')
    plt.ylabel('Số lượng bài hát')
    plt.title('Phân bố bài hát theo thể loại')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Lưu hoặc hiển thị biểu đồ
    if output_file:
        plt.savefig(output_file)
        print(f"Đã lưu biểu đồ vào file: {output_file}")
    else:
        plt.show()

def visualize_feature_distribution(db_path, feature_name, output_file=None):
    """
    Trực quan hóa phân bố của một đặc trưng trong cơ sở dữ liệu
    
    Parameters:
    -----------
    db_path : str
        Đường dẫn đến file cơ sở dữ liệu SQLite
    feature_name : str
        Tên đặc trưng cần hiển thị
    output_file : str, optional
        Đường dẫn để lưu file hình ảnh
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Lấy tất cả đặc trưng
    cursor.execute("SELECT features FROM songs")
    rows = cursor.fetchall()
    
    conn.close()
    
    if not rows:
        print("Không có dữ liệu đặc trưng để hiển thị.")
        return
    
    # Trích xuất giá trị đặc trưng từ mỗi bài hát
    feature_values = []
    for row in rows:
        try:
            features = joblib.load(io.BytesIO(row[0]))
            if feature_name in features:
                feature_values.append(features[feature_name])
        except:
            continue
    
    if not feature_values:
        print(f"Không tìm thấy đặc trưng '{feature_name}' trong cơ sở dữ liệu.")
        return
    
    # Tạo biểu đồ histogram
    plt.figure(figsize=(10, 6))
    plt.hist(feature_values, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel(feature_name)
    plt.ylabel('Số lượng bài hát')
    plt.title(f'Phân bố đặc trưng "{feature_name}"')
    plt.tight_layout()
    
    # Lưu hoặc hiển thị biểu đồ
    if output_file:
        plt.savefig(output_file)
        print(f"Đã lưu biểu đồ vào file: {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Xem và hiển thị thông tin trong cơ sở dữ liệu nhạc')
    parser.add_argument('--db', default='./database/music_features.db', help='Đường dẫn đến file cơ sở dữ liệu (mặc định: ./database/music_features.db)')
    
    subparsers = parser.add_subparsers(dest='command', help='Các lệnh')
    
    # Lệnh liệt kê tất cả bài hát
    list_parser = subparsers.add_parser('list', help='Liệt kê bài hát')
    list_parser.add_argument('--limit', '-l', type=int, help='Số lượng bài hát tối đa')
    list_parser.add_argument('--offset', '-o', type=int, default=0, help='Vị trí bắt đầu')
    list_parser.add_argument('--sort', '-s', choices=['id', 'title', 'artist', 'genre'], default='id', help='Sắp xếp theo')
    list_parser.add_argument('--genre', '-g', help='Lọc theo thể loại')
    
    # Lệnh hiển thị thống kê
    stats_parser = subparsers.add_parser('stats', help='Hiển thị thống kê')
    
    # Lệnh trực quan hóa phân bố thể loại
    genre_viz_parser = subparsers.add_parser('viz-genre', help='Trực quan hóa phân bố thể loại')
    genre_viz_parser.add_argument('--output', '-o', help='Đường dẫn để lưu file hình ảnh')
    
    # Lệnh trực quan hóa phân bố đặc trưng
    feature_viz_parser = subparsers.add_parser('viz-feature', help='Trực quan hóa phân bố đặc trưng')
    feature_viz_parser.add_argument('feature', choices=['rms_energy', 'zcr', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'tempo'], help='Đặc trưng cần hiển thị')
    feature_viz_parser.add_argument('--output', '-o', help='Đường dẫn để lưu file hình ảnh')
    
    args = parser.parse_args()
    
    # Kiểm tra database tồn tại
    if not os.path.exists(args.db):
        print(f"Lỗi: Không tìm thấy file cơ sở dữ liệu tại {args.db}")
        return
    
    # Xử lý các lệnh
    if args.command == 'list':
        songs = list_all_songs(args.db, args.limit, args.offset, args.sort, args.genre)
        print_song_table(songs)
        print(f"Hiển thị {len(songs)} bài hát, bắt đầu từ vị trí {args.offset}")
    
    elif args.command == 'stats':
        stats = get_database_stats(args.db)
        
        print(f"Tổng số bài hát: {stats['total_songs']}")
        
        if stats['genre_stats']:
            print("\nPhân bố theo thể loại:")
            for genre, count in stats['genre_stats']:
                print(f"  - {genre}: {count} bài hát")
        
        if stats['artist_stats']:
            print("\nTop nghệ sĩ:")
            for artist, count in stats['artist_stats']:
                print(f"  - {artist}: {count} bài hát")
    
    elif args.command == 'viz-genre':
        visualize_genre_distribution(args.db, args.output)
    
    elif args.command == 'viz-feature':
        visualize_feature_distribution(args.db, args.feature, args.output)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 