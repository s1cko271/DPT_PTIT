#!/usr/bin/env python3
"""
HỆ THỐNG TÌM KIẾM ÂM NHẠC DỰA TRÊN ĐẶC TRƯNG ÂM THANH
=====================================================

Script tổng thể để quản lý hệ thống tìm kiếm âm nhạc,
tích hợp tất cả các thành phần:
- Xây dựng cơ sở dữ liệu
- Tìm kiếm bài hát tương tự
- Trực quan hóa kết quả
"""

import os
import argparse
import time
from modified_feature_extraction import extract_and_save_features
from audio_processing_pipeline import AudioProcessingPipeline
from visualize_results import visualize_pipeline_results

def print_header(text):
    """In tiêu đề với định dạng đẹp"""
    print("\n" + "=" * 80)
    print(f" {text} ".center(80, "="))
    print("=" * 80)

def check_database(db_path):
    """Kiểm tra cơ sở dữ liệu và thông tin số lượng bài hát"""
    import sqlite3
    
    if not os.path.exists(db_path):
        print(f"Cơ sở dữ liệu không tồn tại: {db_path}")
        return 0
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM songs")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        print(f"Lỗi khi kiểm tra cơ sở dữ liệu: {e}")
        return 0

def build_database(audio_dir, db_path, force_rebuild=False):
    """Xây dựng cơ sở dữ liệu từ thư mục âm thanh"""
    print_header("XÂY DỰNG CƠ SỞ DỮ LIỆU")
    
    # Kiểm tra xem cơ sở dữ liệu đã tồn tại chưa
    if os.path.exists(db_path) and not force_rebuild:
        song_count = check_database(db_path)
        if song_count > 0:
            print(f"Cơ sở dữ liệu đã tồn tại với {song_count} bài hát tại: {db_path}")
            choice = input("Bạn có muốn xây dựng lại không? (y/n): ")
            if choice.lower() != 'y':
                print("Giữ nguyên cơ sở dữ liệu hiện tại.")
                return
    
    # Đảm bảo thư mục chứa file database tồn tại
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    
    # Nếu force_rebuild và file tồn tại, xóa file cũ
    if os.path.exists(db_path) and force_rebuild:
        try:
            os.remove(db_path)
            print(f"Đã xóa cơ sở dữ liệu cũ: {db_path}")
        except Exception as e:
            print(f"Lỗi khi xóa cơ sở dữ liệu cũ: {e}")
    
    # Kiểm tra thư mục âm thanh
    if not os.path.exists(audio_dir):
        print(f"Lỗi: Thư mục âm thanh không tồn tại: {audio_dir}")
        return
    
    # Đếm số lượng file âm thanh
    audio_files = []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(('.mp3', '.wav', '.ogg')):
                audio_files.append(os.path.join(root, file))
    
    if not audio_files:
        print(f"Lỗi: Không tìm thấy file âm thanh trong thư mục: {audio_dir}")
        return
    
    print(f"Đã tìm thấy {len(audio_files)} file âm thanh trong thư mục: {audio_dir}")
    print(f"Bắt đầu trích xuất đặc trưng và xây dựng cơ sở dữ liệu...")
    
    # Đo thời gian xây dựng
    start_time = time.time()
    
    # Trích xuất đặc trưng và lưu vào CSDL
    extract_and_save_features(audio_dir, db_path)
    
    # Kiểm tra kết quả
    song_count = check_database(db_path)
    elapsed_time = time.time() - start_time
    
    print(f"\nHoàn thành xây dựng cơ sở dữ liệu với {song_count} bài hát.")
    print(f"Thời gian xây dựng: {elapsed_time:.2f} giây")
    print(f"Cơ sở dữ liệu được lưu tại: {os.path.abspath(db_path)}")

def search_music(audio_path, db_path, similarity_method="cosine", top_k=3, visualize=False, save_path=None):
    """Tìm kiếm bài hát tương tự với file âm thanh đầu vào"""
    print_header("TÌM KIẾM ÂM NHẠC TƯƠNG TỰ")
    
    # Kiểm tra file âm thanh
    if not os.path.exists(audio_path):
        print(f"Lỗi: File âm thanh không tồn tại: {audio_path}")
        return
    
    # Kiểm tra cơ sở dữ liệu
    song_count = check_database(db_path)
    if song_count == 0:
        print(f"Lỗi: Cơ sở dữ liệu trống hoặc không tồn tại: {db_path}")
        return
    
    print(f"File âm thanh: {os.path.basename(audio_path)}")
    print(f"Cơ sở dữ liệu: {db_path} ({song_count} bài hát)")
    print(f"Phương pháp so sánh: {similarity_method}")
    print(f"Số lượng kết quả: {top_k}")
    
    if visualize:
        # Sử dụng module trực quan hóa
        result = visualize_pipeline_results(
            audio_path, db_path, similarity_method, top_k, save_path
        )
    else:
        # Sử dụng pipeline thông thường
        pipeline = AudioProcessingPipeline(
            database_path=db_path,
            similarity_method=similarity_method
        )
        result = pipeline.process(audio_path, top_k=top_k)
    
    return result

def main():
    """Hàm chính để chạy hệ thống"""
    parser = argparse.ArgumentParser(description="Hệ thống tìm kiếm âm nhạc dựa trên đặc trưng âm thanh")
    subparsers = parser.add_subparsers(dest="command", help="Các lệnh")
    
    # Lệnh xây dựng cơ sở dữ liệu
    build_parser = subparsers.add_parser("build", help="Xây dựng cơ sở dữ liệu từ thư mục âm thanh")
    build_parser.add_argument("--dir", "-d", required=True, help="Thư mục chứa các file âm thanh")
    build_parser.add_argument("--db", default="./database/music_features.db", 
                             help="Đường dẫn đến file cơ sở dữ liệu (mặc định: ./database/music_features.db)")
    build_parser.add_argument("--force", "-f", action="store_true", 
                             help="Xóa cơ sở dữ liệu cũ nếu đã tồn tại")
    
    # Lệnh tìm kiếm bài hát tương tự
    search_parser = subparsers.add_parser("search", help="Tìm kiếm bài hát tương tự với file âm thanh")
    search_parser.add_argument("--file", "-f", required=True, help="Đường dẫn đến file âm thanh cần tìm kiếm")
    search_parser.add_argument("--db", default="./database/music_features.db", 
                              help="Đường dẫn đến file cơ sở dữ liệu (mặc định: ./database/music_features.db)")
    search_parser.add_argument("--method", "-m", choices=["cosine", "euclidean"], default="cosine", 
                              help="Phương pháp tính độ tương đồng (mặc định: cosine)")
    search_parser.add_argument("--top", "-t", type=int, default=3, 
                              help="Số lượng kết quả trả về (mặc định: 3)")
    search_parser.add_argument("--visualize", "-v", action="store_true", 
                              help="Trực quan hóa kết quả tìm kiếm")
    search_parser.add_argument("--save", "-s", 
                              help="Thư mục để lưu biểu đồ kết quả trực quan")
    
    # Lệnh demo toàn bộ hệ thống
    demo_parser = subparsers.add_parser("demo", help="Demo toàn bộ hệ thống từ đầu đến cuối")
    demo_parser.add_argument("--dir", "-d", default="./songs", 
                            help="Thư mục chứa các file âm thanh (mặc định: ./songs)")
    demo_parser.add_argument("--db", default="./database/music_features.db", 
                            help="Đường dẫn đến file cơ sở dữ liệu (mặc định: ./database/music_features.db)")
    demo_parser.add_argument("--file", "-f", required=True, 
                            help="Đường dẫn đến file âm thanh cần tìm kiếm")
    demo_parser.add_argument("--method", "-m", choices=["cosine", "euclidean"], default="cosine", 
                            help="Phương pháp tính độ tương đồng (mặc định: cosine)")
    demo_parser.add_argument("--rebuild", "-r", action="store_true", 
                            help="Xây dựng lại cơ sở dữ liệu")
    
    args = parser.parse_args()
    
    # Xử lý các lệnh
    if args.command == "build":
        build_database(args.dir, args.db, args.force)
    
    elif args.command == "search":
        search_music(args.file, args.db, args.method, args.top, args.visualize, args.save)
    
    elif args.command == "demo":
        print_header("DEMO HỆ THỐNG TÌM KIẾM ÂM NHẠC")
        
        # Nếu cần, xây dựng lại cơ sở dữ liệu
        if args.rebuild or not os.path.exists(args.db):
            build_database(args.dir, args.db, args.rebuild)
        
        # Tìm kiếm và trực quan hóa
        search_music(args.file, args.db, args.method, 3, True, "./results")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 