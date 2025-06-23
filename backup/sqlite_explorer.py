#!/usr/bin/env python3
"""
SQLITE EXPLORER
-------------
Script để tương tác trực tiếp với cơ sở dữ liệu SQLite sử dụng các câu truy vấn SQL tùy chỉnh
"""

import os
import sqlite3
import argparse
import io
import joblib
import numpy as np
from tabulate import tabulate

def execute_query(db_path, query, params=None, fetch=True):
    """
    Thực thi một câu truy vấn SQL tùy chỉnh
    
    Parameters:
    -----------
    db_path : str
        Đường dẫn đến file cơ sở dữ liệu SQLite
    query : str
        Câu truy vấn SQL
    params : tuple, optional
        Tham số cho câu truy vấn
    fetch : bool
        Có lấy kết quả không
        
    Returns:
    --------
    results : list
        Kết quả truy vấn
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        if fetch:
            results = cursor.fetchall()
            conn.commit()
            conn.close()
            return results
        else:
            conn.commit()
            conn.close()
            return None
    except sqlite3.Error as e:
        conn.close()
        raise e

def get_table_info(db_path, table_name=None):
    """
    Lấy thông tin về các bảng trong cơ sở dữ liệu
    
    Parameters:
    -----------
    db_path : str
        Đường dẫn đến file cơ sở dữ liệu SQLite
    table_name : str, optional
        Tên bảng cụ thể để lấy thông tin, nếu None sẽ lấy thông tin tất cả các bảng
        
    Returns:
    --------
    tables_info : dict
        Thông tin về các bảng
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Lấy danh sách các bảng
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    if table_name and table_name not in tables:
        conn.close()
        raise ValueError(f"Không tìm thấy bảng '{table_name}' trong cơ sở dữ liệu")
    
    tables_info = {}
    
    # Nếu chỉ định tên bảng, chỉ lấy thông tin của bảng đó
    if table_name:
        tables = [table_name]
    
    for table in tables:
        # Lấy thông tin về cấu trúc bảng
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        
        # Lấy số lượng bản ghi trong bảng
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        
        tables_info[table] = {
            'columns': columns,
            'count': count
        }
    
    conn.close()
    return tables_info

def print_table_info(tables_info):
    """
    In thông tin về các bảng
    
    Parameters:
    -----------
    tables_info : dict
        Thông tin về các bảng
    """
    for table_name, info in tables_info.items():
        print(f"\nBảng: {table_name} ({info['count']} bản ghi)")
        
        # In thông tin về các cột
        headers = ["#", "Tên", "Kiểu dữ liệu", "NotNull", "Mặc định", "Khóa chính"]
        rows = []
        
        for col in info['columns']:
            rows.append([
                col[0],          # cid
                col[1],          # name
                col[2],          # type
                "Có" if col[3] else "Không",  # notnull
                col[4] if col[4] is not None else "",  # dflt_value
                "Có" if col[5] else "Không"   # pk
            ])
        
        print(tabulate(rows, headers=headers, tablefmt="grid"))

def print_query_results(results, headers=None):
    """
    In kết quả của câu truy vấn
    
    Parameters:
    -----------
    results : list
        Kết quả truy vấn
    headers : list, optional
        Tiêu đề các cột
    """
    if not results:
        print("Không có kết quả nào được trả về.")
        return
    
    # Nếu không có headers được cung cấp, sử dụng số thứ tự
    if headers is None:
        headers = [f"Cột {i+1}" for i in range(len(results[0]))]
    
    print(tabulate(results, headers=headers, tablefmt="grid"))
    print(f"Tổng số kết quả: {len(results)}")

def dump_blob_data(db_path, table, blob_column, id_column, id_value, output_file):
    """
    Xuất dữ liệu BLOB từ cơ sở dữ liệu
    
    Parameters:
    -----------
    db_path : str
        Đường dẫn đến file cơ sở dữ liệu SQLite
    table : str
        Tên bảng
    blob_column : str
        Tên cột chứa dữ liệu BLOB
    id_column : str
        Tên cột ID
    id_value : str
        Giá trị ID
    output_file : str
        Đường dẫn đến file đầu ra
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute(f"SELECT {blob_column} FROM {table} WHERE {id_column} = ?", (id_value,))
        result = cursor.fetchone()
        
        if result is None:
            print(f"Không tìm thấy bản ghi có {id_column} = {id_value}")
            return
        
        blob_data = result[0]
        
        # Lưu dữ liệu BLOB vào file
        with open(output_file, 'wb') as f:
            f.write(blob_data)
        
        print(f"Đã xuất dữ liệu BLOB vào file: {output_file}")
    except sqlite3.Error as e:
        print(f"Lỗi khi xuất dữ liệu BLOB: {e}")
    finally:
        conn.close()

def decode_features_blob(db_path, id_value):
    """
    Giải mã BLOB đặc trưng âm thanh từ cơ sở dữ liệu
    
    Parameters:
    -----------
    db_path : str
        Đường dẫn đến file cơ sở dữ liệu SQLite
    id_value : int
        ID của bài hát
        
    Returns:
    --------
    features : dict
        Đặc trưng âm thanh
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT features FROM songs WHERE id = ?", (id_value,))
        result = cursor.fetchone()
        
        if result is None:
            print(f"Không tìm thấy bài hát có ID = {id_value}")
            return None
        
        blob_data = result[0]
        
        # Giải mã đặc trưng
        features = joblib.load(io.BytesIO(blob_data))
        return features
    except sqlite3.Error as e:
        print(f"Lỗi khi giải mã đặc trưng: {e}")
        return None
    finally:
        conn.close()

def interactive_mode(db_path):
    """
    Chế độ tương tác với cơ sở dữ liệu
    
    Parameters:
    -----------
    db_path : str
        Đường dẫn đến file cơ sở dữ liệu SQLite
    """
    print(f"Kết nối đến cơ sở dữ liệu: {db_path}")
    
    while True:
        print("\n=== SQLITE EXPLORER ===")
        print("1. Xem thông tin cơ sở dữ liệu")
        print("2. Thực thi truy vấn SQL")
        print("3. Xuất dữ liệu BLOB")
        print("4. Giải mã đặc trưng âm thanh")
        print("0. Thoát")
        
        choice = input("\nNhập lựa chọn của bạn: ")
        
        if choice == '0':
            break
        
        # Xem thông tin cơ sở dữ liệu
        elif choice == '1':
            try:
                table_name = input("Nhập tên bảng (Enter để xem tất cả): ")
                table_name = table_name if table_name else None
                
                tables_info = get_table_info(db_path, table_name)
                print_table_info(tables_info)
            except Exception as e:
                print(f"Lỗi: {e}")
        
        # Thực thi truy vấn SQL
        elif choice == '2':
            query = input("Nhập câu truy vấn SQL: ")
            
            try:
                if query.lower().startswith(('select', 'pragma', 'explain')):
                    results = execute_query(db_path, query)
                    print_query_results(results)
                else:
                    execute_query(db_path, query, fetch=False)
                    print("Truy vấn đã được thực thi thành công.")
            except Exception as e:
                print(f"Lỗi: {e}")
        
        # Xuất dữ liệu BLOB
        elif choice == '3':
            try:
                table = input("Nhập tên bảng: ")
                blob_column = input("Nhập tên cột BLOB: ")
                id_column = input("Nhập tên cột ID: ")
                id_value = input("Nhập giá trị ID: ")
                output_file = input("Nhập đường dẫn file đầu ra: ")
                
                dump_blob_data(db_path, table, blob_column, id_column, id_value, output_file)
            except Exception as e:
                print(f"Lỗi: {e}")
        
        # Giải mã đặc trưng âm thanh
        elif choice == '4':
            try:
                id_value = input("Nhập ID bài hát: ")
                
                if not id_value.isdigit():
                    print("ID không hợp lệ. Vui lòng nhập một số nguyên.")
                    continue
                
                features = decode_features_blob(db_path, int(id_value))
                
                if features:
                    print("\nĐặc trưng âm thanh:")
                    for key, value in features.items():
                        if key == 'feature_vector':
                            print(f"  - {key}: {value}")
                        else:
                            print(f"  - {key}: {value}")
            except Exception as e:
                print(f"Lỗi: {e}")
        
        else:
            print("Lựa chọn không hợp lệ. Vui lòng thử lại.")

def main():
    parser = argparse.ArgumentParser(description='Tương tác trực tiếp với cơ sở dữ liệu SQLite')
    parser.add_argument('--db', default='./database/music_features.db', help='Đường dẫn đến file cơ sở dữ liệu (mặc định: ./database/music_features.db)')
    parser.add_argument('--query', '-q', help='Câu truy vấn SQL để thực thi')
    parser.add_argument('--table', '-t', help='Tên bảng để xem thông tin')
    parser.add_argument('--dump-blob', '-d', nargs=5, metavar=('TABLE', 'BLOB_COLUMN', 'ID_COLUMN', 'ID_VALUE', 'OUTPUT_FILE'), help='Xuất dữ liệu BLOB từ cơ sở dữ liệu')
    parser.add_argument('--decode-features', '-f', type=int, metavar='ID', help='Giải mã đặc trưng âm thanh từ cơ sở dữ liệu')
    parser.add_argument('--interactive', '-i', action='store_true', help='Chế độ tương tác')
    
    args = parser.parse_args()
    
    # Kiểm tra database tồn tại
    if not os.path.exists(args.db):
        print(f"Lỗi: Không tìm thấy file cơ sở dữ liệu tại {args.db}")
        return
    
    # Xử lý các lệnh
    if args.query:
        try:
            if args.query.lower().startswith(('select', 'pragma', 'explain')):
                results = execute_query(args.db, args.query)
                print_query_results(results)
            else:
                execute_query(args.db, args.query, fetch=False)
                print("Truy vấn đã được thực thi thành công.")
        except Exception as e:
            print(f"Lỗi: {e}")
    
    elif args.table:
        try:
            tables_info = get_table_info(args.db, args.table)
            print_table_info(tables_info)
        except Exception as e:
            print(f"Lỗi: {e}")
    
    elif args.dump_blob:
        try:
            table, blob_column, id_column, id_value, output_file = args.dump_blob
            dump_blob_data(args.db, table, blob_column, id_column, id_value, output_file)
        except Exception as e:
            print(f"Lỗi: {e}")
    
    elif args.decode_features:
        try:
            features = decode_features_blob(args.db, args.decode_features)
            
            if features:
                print("\nĐặc trưng âm thanh:")
                for key, value in features.items():
                    if key == 'feature_vector':
                        print(f"  - {key}: {value}")
                    else:
                        print(f"  - {key}: {value}")
        except Exception as e:
            print(f"Lỗi: {e}")
    
    elif args.interactive:
        interactive_mode(args.db)
    
    else:
        interactive_mode(args.db)  # Mặc định là chế độ tương tác

if __name__ == "__main__":
    main() 