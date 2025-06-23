import os
import argparse
import time
from feature_extraction import extract_features, search_similar_songs, extract_and_save_features
import sqlite3

def search_by_file(db_path, query_file, num_results=3):
    """
    Tìm kiếm nhạc tương tự dựa trên file audio đầu vào sử dụng Cosine Similarity
    
    Parameters:
    -----------
    db_path : str
        Đường dẫn đến file cơ sở dữ liệu SQLite
    query_file : str
        Đường dẫn đến file âm thanh cần tìm kiếm
    num_results : int, optional
        Số lượng kết quả cần trả về (mặc định: 3)
    
    Returns:
    --------
    similar_songs : list
        Danh sách các bài hát tương tự
    """
    start_time = time.time()
    
    # Trích xuất đặc trưng từ file truy vấn
    print(f"Đang trích xuất đặc trưng từ file: {query_file}")
    query_features = extract_features(query_file)
    
    if query_features is None:
        print(f"Không thể trích xuất đặc trưng từ file: {query_file}")
        return []
    
    # Tìm kiếm các bài hát tương tự
    print(f"Đang tìm kiếm bài hát tương tự...")
    similar_songs = search_similar_songs(
        db_path, 
        query_features, 
        num_results=num_results
    )
    
    # Tính thời gian thực hiện
    elapsed_time = time.time() - start_time
    print(f"Thời gian tìm kiếm: {elapsed_time:.2f} giây")
    
    # Hiển thị kết quả
    print(f"\nKết quả tìm kiếm ({len(similar_songs)} bài hát):")
    print("-" * 80)
    
    for i, song in enumerate(similar_songs):
        sim_value = f"{song['similarity']:.4f} (càng cao càng giống)"
            
        print(f"{i+1}. {song['title']} - {song['artist']}")
        print(f"   Filename: {song['filename']}")
        print(f"   Độ tương đồng: {sim_value}")
        print("-" * 80)
    
    return similar_songs

def rebuild_database(audio_dir, db_path):
    """
    Xây dựng lại cơ sở dữ liệu từ thư mục nhạc
    """
    # Xóa cơ sở dữ liệu cũ nếu tồn tại
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Đã xóa cơ sở dữ liệu cũ: {db_path}")
    
    # Tạo và điền dữ liệu vào cơ sở dữ liệu mới
    print(f"Đang xây dựng cơ sở dữ liệu mới từ thư mục: {audio_dir}")
    extract_and_save_features(audio_dir, db_path)
    
    # Hiển thị thông tin về cơ sở dữ liệu
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM songs")
    count = cursor.fetchone()[0]
    conn.close()
    
    print(f"Đã xây dựng cơ sở dữ liệu với {count} bài hát.")

def main():
    parser = argparse.ArgumentParser(description='Music Search System')
    parser.add_argument('action', choices=['build', 'search'], help='Action to perform')
    parser.add_argument('path', help='Path to music file or directory')
    parser.add_argument('--db', default=os.path.join('database', 'music_features.db'), help='Path to database file')
    parser.add_argument('--num', type=int, default=3, help='Number of similar songs to return')
    
    args = parser.parse_args()
    
    if args.action == 'build':
        print(f"Building database from {args.path}...")
        extract_and_save_features(args.path, args.db)
        print("Database built successfully!")
    
    elif args.action == 'search':
        print(f"Searching for songs similar to {args.path}...")
        query_features = extract_features(args.path)
        if query_features is None:
            print("Error: Could not extract features from the query file.")
            return
        
        similar_songs = search_similar_songs(args.db, query_features, args.num)
        
        print("\nTop similar songs:")
        for i, song in enumerate(similar_songs, 1):
            print(f"\n{i}. {song['title']} - {song['artist']}")
            print(f"   Album: {song['album']}")
            print(f"   Similarity: {song['similarity']:.2f}")

if __name__ == '__main__':
    main() 