import os
import numpy as np
from modified_feature_extraction import extract_and_save_features, retrieve_features, get_feature_vectors

# Đường dẫn đến thư mục chứa file nhạc và cơ sở dữ liệu
AUDIO_DIR = "./songs"  # Thay đổi thành thư mục chứa file nhạc của bạn
DB_PATH = "./music_features.db"

def main():
    # 1. Trích xuất và lưu đặc trưng vào cơ sở dữ liệu
    if not os.path.exists(DB_PATH) or input("Cơ sở dữ liệu đã tồn tại. Tạo lại? (y/n): ").lower() == 'y':
        print("Đang trích xuất đặc trưng và lưu vào cơ sở dữ liệu...")
        extract_and_save_features(AUDIO_DIR, DB_PATH)
    
    # 2. Lấy thông tin và đặc trưng của một bài hát
    song_filename = input("Nhập tên file bài hát (ví dụ: song.mp3): ")
    song_info = retrieve_features(DB_PATH, filename=song_filename)
    
    if song_info:
        print("\nThông tin bài hát:")
        print(f"ID: {song_info['id']}")
        print(f"Tên: {song_info['title']}")
        print(f"Ca sĩ: {song_info['artist']}")
        print(f"Album: {song_info['album']}")
        print("\nĐặc trưng âm thanh:")
        for feature_name, value in song_info['features'].items():
            if feature_name != 'feature_vector':
                print(f"- {feature_name}: {value}")
        
        print("\nVector đặc trưng:")
        print(song_info['features']['feature_vector'])
    else:
        print(f"Không tìm thấy bài hát '{song_filename}' trong cơ sở dữ liệu.")
    
    # 3. Truy vấn tất cả vector đặc trưng
    print("\nĐang lấy danh sách vector đặc trưng từ cơ sở dữ liệu...")
    all_features = get_feature_vectors(DB_PATH)
    print(f"Tìm thấy {len(all_features)} bài hát trong cơ sở dữ liệu.")
    
    # 4. Tìm bài hát có đặc trưng tương tự
    print("\nBài hát có đặc trưng tương tự:")
    if song_info and all_features:
        target_vector = song_info['features']['feature_vector']
        
        # Tính khoảng cách giữa vector đặc trưng
        distances = []
        for song in all_features:
            if song['id'] != song_info['id']:  # Không so sánh với chính nó
                distance = np.linalg.norm(target_vector - song['feature_vector'])
                distances.append((song, distance))
        
        # Sắp xếp theo khoảng cách tăng dần
        distances.sort(key=lambda x: x[1])
        
        # Hiển thị 5 bài hát gần nhất
        for i, (song, distance) in enumerate(distances[:5]):
            print(f"{i+1}. {song['title']} (Ca sĩ: {song['artist']}) - Khoảng cách: {distance:.4f}")

if __name__ == "__main__":
    main() 