#!/usr/bin/env python3
"""
EXPLORE DATABASE
--------------
Script để khám phá và tương tác với cơ sở dữ liệu nhạc
"""

import os
import sqlite3
import argparse
import io
import joblib
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from scipy.io import wavfile
from IPython.display import Audio, display
import tempfile
import subprocess

def get_song_path(db_path, song_id):
    """
    Lấy đường dẫn đến file âm thanh dựa trên ID trong database
    
    Parameters:
    -----------
    db_path : str
        Đường dẫn đến file cơ sở dữ liệu SQLite
    song_id : int
        ID của bài hát
        
    Returns:
    --------
    path : str
        Đường dẫn đến file âm thanh
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT filename FROM songs WHERE id = ?", (song_id,))
    result = cursor.fetchone()
    
    conn.close()
    
    if result:
        # Giả sử cấu trúc là audio_files/genre/filename
        return os.path.join("audio_files", result[0])
    
    return None

def play_song(audio_path):
    """
    Phát một bài hát
    
    Parameters:
    -----------
    audio_path : str
        Đường dẫn đến file âm thanh
    """
    if not os.path.exists(audio_path):
        print(f"Không tìm thấy file âm thanh: {audio_path}")
        return
    
    # Sử dụng player có sẵn trong hệ thống
    try:
        if os.name == 'nt':  # Windows
            os.startfile(audio_path)
        elif os.name == 'posix':  # Linux, macOS
            subprocess.call(('xdg-open', audio_path))
        print(f"Đang phát: {os.path.basename(audio_path)}")
    except:
        print(f"Không thể phát file: {audio_path}")

def compare_features(db_path, song_id1, song_id2):
    """
    So sánh đặc trưng giữa hai bài hát
    
    Parameters:
    -----------
    db_path : str
        Đường dẫn đến file cơ sở dữ liệu SQLite
    song_id1, song_id2 : int
        ID của hai bài hát cần so sánh
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Lấy thông tin của hai bài hát
    cursor.execute("""
    SELECT id, filename, title, artist, genre, features 
    FROM songs 
    WHERE id IN (?, ?)
    """, (song_id1, song_id2))
    
    results = cursor.fetchall()
    conn.close()
    
    if len(results) < 2:
        print("Không tìm thấy đủ hai bài hát để so sánh.")
        return
    
    # Trích xuất thông tin và đặc trưng
    songs = []
    for result in results:
        features = joblib.load(io.BytesIO(result[5]))
        songs.append({
            'id': result[0],
            'filename': result[1],
            'title': result[2],
            'artist': result[3],
            'genre': result[4],
            'features': features
        })
    
    # Sắp xếp lại để song_id1 ở vị trí đầu tiên
    if songs[0]['id'] != song_id1:
        songs = [songs[1], songs[0]]
    
    # So sánh thông tin cơ bản
    print(f"So sánh giữa:")
    print(f"1. {songs[0]['title']} (Nghệ sĩ: {songs[0]['artist']}, Thể loại: {songs[0]['genre']})")
    print(f"2. {songs[1]['title']} (Nghệ sĩ: {songs[1]['artist']}, Thể loại: {songs[1]['genre']})")
    
    print("\nSo sánh đặc trưng âm thanh:")
    
    # Tạo bảng so sánh đặc trưng
    headers = ["Đặc trưng", songs[0]['title'], songs[1]['title'], "Chênh lệch", "Chênh lệch %"]
    rows = []
    
    for feature in ['rms_energy', 'zcr', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'tempo']:
        if feature in songs[0]['features'] and feature in songs[1]['features']:
            val1 = songs[0]['features'][feature]
            val2 = songs[1]['features'][feature]
            diff = val2 - val1
            if val1 != 0:
                diff_percent = (diff / val1) * 100
            else:
                diff_percent = float('inf')
            
            rows.append([
                feature,
                f"{val1:.4f}",
                f"{val2:.4f}",
                f"{diff:.4f}",
                f"{diff_percent:.2f}%"
            ])
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    # Tính độ tương đồng
    vec1 = songs[0]['features']['feature_vector']
    vec2 = songs[1]['features']['feature_vector']
    
    # Chuẩn hóa vector
    vec1_norm = vec1 / np.linalg.norm(vec1) if np.linalg.norm(vec1) > 0 else vec1
    vec2_norm = vec2 / np.linalg.norm(vec2) if np.linalg.norm(vec2) > 0 else vec2
    
    # Tính cosine similarity
    cosine_sim = np.dot(vec1_norm, vec2_norm)
    
    # Tính Euclidean distance
    euclidean_dist = np.linalg.norm(vec1 - vec2)
    
    print(f"\nĐộ tương đồng:")
    print(f"- Cosine similarity: {cosine_sim:.4f} (càng cao càng giống nhau)")
    print(f"- Euclidean distance: {euclidean_dist:.4f} (càng thấp càng giống nhau)")

def visualize_feature_comparison(db_path, genre=None, feature_name='tempo', limit=10):
    """
    Trực quan hóa so sánh một đặc trưng giữa các bài hát
    
    Parameters:
    -----------
    db_path : str
        Đường dẫn đến file cơ sở dữ liệu SQLite
    genre : str, optional
        Lọc theo thể loại
    feature_name : str
        Tên đặc trưng cần so sánh
    limit : int
        Số lượng bài hát tối đa để so sánh
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Tạo câu truy vấn
    query = "SELECT id, title, artist, genre, features FROM songs"
    params = []
    
    if genre:
        query += " WHERE genre LIKE ?"
        params.append(f"%{genre}%")
    
    query += " LIMIT ?"
    params.append(limit)
    
    # Thực thi truy vấn
    cursor.execute(query, params)
    results = cursor.fetchall()
    
    conn.close()
    
    if not results:
        print("Không tìm thấy bài hát phù hợp.")
        return
    
    # Trích xuất dữ liệu đặc trưng
    data = []
    for result in results:
        try:
            features = joblib.load(io.BytesIO(result[4]))
            if feature_name in features:
                data.append({
                    'id': result[0],
                    'title': result[1] or "Unknown",
                    'artist': result[2] or "Unknown",
                    'genre': result[3] or "Unknown",
                    'value': features[feature_name]
                })
        except:
            continue
    
    if not data:
        print(f"Không tìm thấy đặc trưng '{feature_name}' trong các bài hát.")
        return
    
    # Sắp xếp theo giá trị đặc trưng
    data.sort(key=lambda x: x['value'], reverse=True)
    
    # Trích xuất dữ liệu cho biểu đồ
    titles = [f"{d['title']} ({d['artist']})" for d in data]
    values = [d['value'] for d in data]
    genres = [d['genre'] for d in data]
    
    # Tạo màu sắc dựa trên thể loại
    unique_genres = list(set(genres))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_genres)))
    genre_colors = {genre: colors[i] for i, genre in enumerate(unique_genres)}
    bar_colors = [genre_colors[genre] for genre in genres]
    
    # Tạo biểu đồ
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(titles)), values, color=bar_colors)
    
    # Thêm nhãn và tiêu đề
    plt.xlabel('Bài hát')
    plt.ylabel(feature_name)
    plt.title(f'So sánh {feature_name} giữa các bài hát')
    plt.xticks(range(len(titles)), titles, rotation=45, ha='right')
    plt.tight_layout()
    
    # Thêm chú thích
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=genre_colors[g]) for g in unique_genres]
    plt.legend(legend_handles, unique_genres, title="Thể loại")
    
    plt.show()

def interactive_explore(db_path):
    """
    Khám phá tương tác với cơ sở dữ liệu
    
    Parameters:
    -----------
    db_path : str
        Đường dẫn đến file cơ sở dữ liệu SQLite
    """
    while True:
        print("\n=== KHÁM PHÁ CƠ SỞ DỮ LIỆU ===")
        print("1. Liệt kê các bài hát")
        print("2. Tìm kiếm bài hát")
        print("3. Phát một bài hát")
        print("4. So sánh hai bài hát")
        print("5. Trực quan hóa đặc trưng")
        print("0. Thoát")
        
        choice = input("\nNhập lựa chọn của bạn: ")
        
        if choice == '0':
            break
        
        # Liệt kê các bài hát
        elif choice == '1':
            limit = input("Số lượng bài hát (Enter để xem tất cả): ")
            genre = input("Lọc theo thể loại (Enter để bỏ qua): ")
            
            limit = int(limit) if limit.isdigit() else None
            genre = genre if genre else None
            
            # Import function từ view_db.py để liệt kê bài hát
            from view_db import list_all_songs, print_song_table
            songs = list_all_songs(db_path, limit, 0, 'title', genre)
            print_song_table(songs)
        
        # Tìm kiếm bài hát
        elif choice == '2':
            print("\n=== TÌM KIẾM BÀI HÁT ===")
            print("1. Theo tiêu đề")
            print("2. Theo nghệ sĩ")
            print("3. Theo thể loại")
            
            search_choice = input("\nNhập lựa chọn của bạn: ")
            query = input("Nhập từ khóa tìm kiếm: ")
            
            # Import functions từ query_db.py để tìm kiếm
            if search_choice == '1':
                from query_db import search_by_title, print_song_table
                results = search_by_title(db_path, query)
                print_song_table(results)
            elif search_choice == '2':
                from query_db import search_by_artist, print_song_table
                results = search_by_artist(db_path, query)
                print_song_table(results)
            elif search_choice == '3':
                from query_db import search_by_genre, print_song_table
                results = search_by_genre(db_path, query)
                print_song_table(results)
        
        # Phát một bài hát
        elif choice == '3':
            song_id = input("Nhập ID bài hát cần phát: ")
            if song_id.isdigit():
                song_path = get_song_path(db_path, int(song_id))
                if song_path:
                    play_song(song_path)
                else:
                    print(f"Không tìm thấy bài hát có ID = {song_id}")
            else:
                print("ID không hợp lệ. Vui lòng nhập một số nguyên.")
        
        # So sánh hai bài hát
        elif choice == '4':
            id1 = input("Nhập ID bài hát thứ nhất: ")
            id2 = input("Nhập ID bài hát thứ hai: ")
            
            if id1.isdigit() and id2.isdigit():
                compare_features(db_path, int(id1), int(id2))
            else:
                print("ID không hợp lệ. Vui lòng nhập số nguyên.")
        
        # Trực quan hóa đặc trưng
        elif choice == '5':
            print("\n=== TRỰC QUAN HÓA ĐẶC TRƯNG ===")
            print("Chọn đặc trưng cần hiển thị:")
            print("1. rms_energy (Độ to/nhỏ của âm thanh)")
            print("2. zcr (Zero-Crossing Rate)")
            print("3. spectral_centroid (Trọng tâm phổ)")
            print("4. spectral_bandwidth (Độ rộng phổ)")
            print("5. spectral_rolloff (Rolloff phổ)")
            print("6. tempo (Nhịp độ)")
            
            feature_choice = input("\nNhập lựa chọn của bạn: ")
            genre_filter = input("Lọc theo thể loại (Enter để bỏ qua): ")
            
            feature_map = {
                '1': 'rms_energy',
                '2': 'zcr',
                '3': 'spectral_centroid',
                '4': 'spectral_bandwidth',
                '5': 'spectral_rolloff',
                '6': 'tempo'
            }
            
            if feature_choice in feature_map:
                feature = feature_map[feature_choice]
                genre = genre_filter if genre_filter else None
                visualize_feature_comparison(db_path, genre, feature)
            else:
                print("Lựa chọn không hợp lệ.")
        
        else:
            print("Lựa chọn không hợp lệ. Vui lòng thử lại.")

def main():
    parser = argparse.ArgumentParser(description='Khám phá và tương tác với cơ sở dữ liệu nhạc')
    parser.add_argument('--db', default='./database/music_features.db', help='Đường dẫn đến file cơ sở dữ liệu (mặc định: ./database/music_features.db)')
    parser.add_argument('--compare', '-c', nargs=2, type=int, metavar=('ID1', 'ID2'), help='So sánh hai bài hát bằng ID')
    parser.add_argument('--play', '-p', type=int, metavar='ID', help='Phát một bài hát theo ID')
    parser.add_argument('--visualize', '-v', choices=['rms_energy', 'zcr', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'tempo'], help='Trực quan hóa một đặc trưng')
    parser.add_argument('--genre', '-g', help='Lọc theo thể loại khi trực quan hóa')
    parser.add_argument('--interactive', '-i', action='store_true', help='Chế độ khám phá tương tác')
    
    args = parser.parse_args()
    
    # Kiểm tra database tồn tại
    if not os.path.exists(args.db):
        print(f"Lỗi: Không tìm thấy file cơ sở dữ liệu tại {args.db}")
        return
    
    # Xử lý các lệnh
    if args.compare:
        compare_features(args.db, args.compare[0], args.compare[1])
    
    elif args.play:
        audio_path = get_song_path(args.db, args.play)
        if audio_path:
            play_song(audio_path)
        else:
            print(f"Không tìm thấy bài hát có ID = {args.play}")
    
    elif args.visualize:
        visualize_feature_comparison(args.db, args.genre, args.visualize)
    
    elif args.interactive:
        interactive_explore(args.db)
    
    else:
        interactive_explore(args.db)  # Mặc định là chế độ tương tác

if __name__ == "__main__":
    main() 