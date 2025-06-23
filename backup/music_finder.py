import os
import argparse
import numpy as np
from modified_feature_extraction import extract_features, extract_and_save_features
from audio_search import search_similar_songs

def build_database(audio_dir, db_path):
    """Xây dựng cơ sở dữ liệu âm thanh từ thư mục"""
    print(f"Đang xây dựng cơ sở dữ liệu từ thư mục '{audio_dir}'...")
    extract_and_save_features(audio_dir, db_path)
    print(f"Hoàn thành xây dựng cơ sở dữ liệu tại '{db_path}'")

def find_similar_music(input_audio, db_path, metric='both', top_k=3):
    """Tìm kiếm bài hát tương tự từ cơ sở dữ liệu"""
    if not os.path.exists(input_audio):
        print(f"Lỗi: Không tìm thấy file '{input_audio}'")
        return
        
    if not os.path.exists(db_path):
        print(f"Lỗi: Không tìm thấy cơ sở dữ liệu '{db_path}'")
        return
        
    print(f"Đang tìm kiếm bài hát tương tự với '{os.path.basename(input_audio)}'...")
    
    if metric == 'both' or metric == 'cosine':
        print("\n=== Kết quả sử dụng Cosine similarity ===")
        cosine_results = search_similar_songs(input_audio, db_path, 'cosine', top_k)
        for i, result in enumerate(cosine_results):
            print(f"{i+1}. {result['title']} (Ca sĩ: {result['artist']})")
            print(f"   Độ tương đồng: {result['similarity']:.4f}")
    
    if metric == 'both' or metric == 'euclidean':
        print("\n=== Kết quả sử dụng Euclidean distance ===")
        euclidean_results = search_similar_songs(input_audio, db_path, 'euclidean', top_k)
        for i, result in enumerate(euclidean_results):
            print(f"{i+1}. {result['title']} (Ca sĩ: {result['artist']})")
            print(f"   Khoảng cách: {result['distance']:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Ứng dụng tìm kiếm bài hát tương tự dựa trên đặc trưng âm thanh')
    
    subparsers = parser.add_subparsers(dest='command', help='Các lệnh')
    
    # Lệnh để xây dựng cơ sở dữ liệu
    build_parser = subparsers.add_parser('build', help='Xây dựng cơ sở dữ liệu từ thư mục âm thanh')
    build_parser.add_argument('--dir', '-d', required=True, help='Thư mục chứa các file âm thanh')
    build_parser.add_argument('--db', default='./music_features.db', help='Đường dẫn đến file cơ sở dữ liệu (mặc định: ./music_features.db)')
    
    # Lệnh để tìm kiếm bài hát tương tự
    search_parser = subparsers.add_parser('search', help='Tìm kiếm bài hát tương tự')
    search_parser.add_argument('--file', '-f', required=True, help='Đường dẫn đến file âm thanh cần tìm kiếm')
    search_parser.add_argument('--db', default='./music_features.db', help='Đường dẫn đến file cơ sở dữ liệu (mặc định: ./music_features.db)')
    search_parser.add_argument('--metric', '-m', choices=['cosine', 'euclidean', 'both'], default='both', 
                             help='Phương pháp tính độ tương đồng (mặc định: both)')
    search_parser.add_argument('--top', '-t', type=int, default=3, help='Số lượng kết quả trả về (mặc định: 3)')
    
    args = parser.parse_args()
    
    if args.command == 'build':
        build_database(args.dir, args.db)
    elif args.command == 'search':
        find_similar_music(args.file, args.db, args.metric, args.top)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 