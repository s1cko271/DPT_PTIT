#!/usr/bin/env python3
"""
Script hướng dẫn chạy demo hệ thống tìm kiếm âm nhạc
"""
import os
import sys
import subprocess
import importlib.util
import shutil

def check_package_installed(package_name):
    """Kiểm tra xem thư viện đã được cài đặt chưa"""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def install_required_packages():
    """Cài đặt các thư viện cần thiết từ requirements.txt"""
    print("Đang cài đặt các thư viện cần thiết...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Cài đặt thư viện thành công!")
    except Exception as e:
        print(f"❌ Lỗi khi cài đặt thư viện: {e}")
        return False
    return True

def check_database():
    """Kiểm tra tồn tại của cơ sở dữ liệu"""
    db_path = "./database/music_features.db"
    if os.path.exists(db_path):
        print(f"✅ Tìm thấy cơ sở dữ liệu: {db_path}")
        
        # Kiểm tra số lượng bài hát trong CSDL
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM songs")
            song_count = cursor.fetchone()[0]
            conn.close()
            print(f"   → CSDL có {song_count} bài hát")
        except:
            print("   → Không thể đọc thông tin từ CSDL")
    else:
        print(f"❌ Không tìm thấy cơ sở dữ liệu: {db_path}")
        print("   → Cần xây dựng CSDL trước khi chạy demo")
        print("   → Chạy lệnh: python music_search_system.py build --dir ./songs")
        return False
    return True

def check_songs_directory():
    """Kiểm tra thư mục songs"""
    songs_dir = "./songs"
    if os.path.exists(songs_dir) and os.path.isdir(songs_dir):
        # Đếm số file âm thanh
        audio_files = []
        for root, _, files in os.walk(songs_dir):
            for file in files:
                if file.endswith(('.mp3', '.wav', '.ogg')):
                    audio_files.append(os.path.join(root, file))
        
        print(f"✅ Tìm thấy thư mục songs với {len(audio_files)} file âm thanh")
    else:
        print(f"❌ Không tìm thấy thư mục songs: {songs_dir}")
        print("   → Cần có thư mục songs chứa các file âm thanh")
        return False
    return True

def run_streamlit_demo():
    """Chạy Streamlit demo"""
    if not check_package_installed("streamlit"):
        print("❌ Chưa cài đặt Streamlit. Đang cài đặt...")
        if not install_required_packages():
            return
    
    print("\n" + "=" * 80)
    print("KHỞI CHẠY DEMO HỆ THỐNG TÌM KIẾM ÂM NHẠC".center(80))
    print("=" * 80)
    
    # Chạy Streamlit sử dụng Python module thay vì lệnh trực tiếp
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_demo.py"])
    except Exception as e:
        print(f"❌ Lỗi khi chạy Streamlit: {e}")
        print("   → Thử chạy lệnh: python -m streamlit run streamlit_demo.py")

def main():
    """Chức năng chính"""
    print("\n" + "=" * 80)
    print(" KIỂM TRA HỆ THỐNG TRƯỚC KHI CHẠY DEMO ".center(80, "="))
    print("=" * 80 + "\n")
    
    # Kiểm tra môi trường
    checks_passed = True
    
    # 1. Kiểm tra các tệp tin cần thiết
    required_files = ["streamlit_demo.py", "audio_processing_pipeline.py", "modified_feature_extraction.py", 
                       "audio_search.py", "visualize_results.py"]
    missing_files = [file for file in required_files if not os.path.exists(file)]
    
    if missing_files:
        print(f"❌ Thiếu các tệp tin: {', '.join(missing_files)}")
        checks_passed = False
    else:
        print("✅ Đã đủ các tệp tin cần thiết")
    
    # 2. Kiểm tra CSDL
    db_exists = check_database()
    if not db_exists:
        checks_passed = False
    
    # 3. Kiểm tra thư mục songs
    songs_exist = check_songs_directory()
    if not songs_exist:
        checks_passed = False
    
    # Nếu tất cả đều OK, chạy demo
    if checks_passed:
        print("\n✅ Tất cả kiểm tra đã thành công! Có thể chạy demo.")
        choice = input("\nBạn có muốn khởi chạy demo ngay bây giờ? (y/n): ")
        if choice.lower() == 'y':
            run_streamlit_demo()
        else:
            print("\nĐể chạy demo, sử dụng lệnh:")
            print("python run_demo.py")
            print("hoặc")
            print("python -m streamlit run streamlit_demo.py")
    else:
        print("\n❌ Kiểm tra không thành công. Vui lòng khắc phục các vấn đề trên trước khi chạy demo.")
        print("\nNếu cần thiết lập cơ sở dữ liệu, sử dụng lệnh:")
        print("python music_search_system.py build --dir ./songs")

if __name__ == "__main__":
    # Kiểm tra nếu được gọi với tham số --run
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        run_streamlit_demo()
    else:
        main() 