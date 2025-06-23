import numpy as np
import librosa
import os
import pandas as pd
import sqlite3
from tqdm import tqdm

def extract_features(file_path, tempo=None):
    """
    Trích xuất đặc trưng từ file âm thanh
    
    Parameters:
    -----------
    file_path : str
        Đường dẫn đến file âm thanh
    tempo : float, optional
        Tempo của file âm thanh (nếu đã biết)
        
    Returns:
    --------
    features : list
        Danh sách các đặc trưng được trích xuất
    """
    # Load file âm thanh
    y, sr = librosa.load(file_path, sr=None)
    
    # Trích xuất tempo nếu chưa được cung cấp
    if tempo is None:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # Tạo frame từ tempo (28 frame được định nghĩa từ tempo)
    frame_length = sr * 60 / tempo
    frames = [int(i * frame_length) for i in range(29)]
    
    # Khởi tạo danh sách để lưu features
    features = []
    
    for i in range(28):
        # Tính thời điểm bắt đầu và kết thúc của mỗi frame
        start = frames[i]
        end = frames[i+1]
        
        # Lấy dữ liệu trong frame
        frame_data = y[start:end]
        
        if len(frame_data) == 0:
            continue
        
        # Trích xuất đặc trưng
        feature_row = [i, i * tempo/60]
        
        # Mới: Thêm root mean square energy
        rms = np.float32(np.sqrt(np.mean(frame_data**2)))
        feature_row.append(rms)
        
        # Mới: Thêm zero crossing rate
        zcr = np.float64(np.mean(librosa.feature.zero_crossing_rate(frame_data)))
        feature_row.append(zcr)
        
        # Spectral centroid
        spectral_centroid = np.float64(np.mean(librosa.feature.spectral_centroid(y=frame_data, sr=sr)))
        feature_row.append(spectral_centroid)
        
        # Mới: Thêm spectral bandwidth
        spectral_bandwidth = np.float64(np.mean(librosa.feature.spectral_bandwidth(y=frame_data, sr=sr)))
        feature_row.append(spectral_bandwidth)
        
        # Mới: Thêm spectral rolloff
        spectral_rolloff = np.float64(np.mean(librosa.feature.spectral_rolloff(y=frame_data, sr=sr)))
        feature_row.append(spectral_rolloff)
        
        # Mới: Thêm MFCCs (4 hệ số đầu tiên)
        mfccs = librosa.feature.mfcc(y=frame_data, sr=sr, n_mfcc=4)
        for mfcc in mfccs:
            feature_row.append(np.float32(np.mean(mfcc)))
        
        # Mới: Thêm chroma features
        chroma = np.float32(np.mean(librosa.feature.chroma_stft(y=frame_data, sr=sr)))
        feature_row.append(chroma)
        
        features.append(feature_row)
    
    return features

def save_features_to_csv(features, output_file='features.csv'):
    """
    Lưu đặc trưng vào file CSV
    
    Parameters:
    -----------
    features : list
        Danh sách các đặc trưng được trích xuất
    output_file : str
        Đường dẫn đến file CSV đầu ra
    """
    # Tạo DataFrame từ features
    columns = [
        'frame', 'time', 'rms', 'zero_crossing_rate', 
        'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
        'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'chroma'
    ]
    
    df = pd.DataFrame(features, columns=columns)
    
    # Lưu vào CSV
    df.to_csv(output_file, index=False)
    
    print(f"Đã lưu features vào {output_file}")
    return df

def update_database(features_df, db_path='audio_features.db'):
    """
    Cập nhật database với các đặc trưng mới
    
    Parameters:
    -----------
    features_df : pandas.DataFrame
        DataFrame chứa các đặc trưng
    db_path : str
        Đường dẫn đến file database
    """
    # Kết nối với database
    conn = sqlite3.connect(db_path)
    
    try:
        # Kiểm tra xem bảng features đã tồn tại chưa
        cursor = conn.cursor()
        cursor.execute('''
        SELECT name FROM sqlite_master WHERE type='table' AND name='features'
        ''')
        
        if cursor.fetchone() is None:
            # Tạo bảng nếu chưa tồn tại
            cursor.execute('''
            CREATE TABLE features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame INTEGER,
                time REAL,
                rms REAL,
                zero_crossing_rate REAL,
                spectral_centroid REAL,
                spectral_bandwidth REAL,
                spectral_rolloff REAL,
                mfcc1 REAL,
                mfcc2 REAL,
                mfcc3 REAL,
                mfcc4 REAL,
                chroma REAL
            )
            ''')
        else:
            # Xóa dữ liệu cũ nếu bảng đã tồn tại
            cursor.execute('DELETE FROM features')
        
        # Lưu DataFrame vào database
        features_df.to_sql('features', conn, if_exists='append', index=False)
        
        print(f"Đã cập nhật database thành công tại {db_path}")
        
    finally:
        conn.close()

def process_audio_directory(directory_path, output_csv='features.csv', db_path='audio_features.db'):
    """
    Xử lý tất cả các file âm thanh trong thư mục
    
    Parameters:
    -----------
    directory_path : str
        Đường dẫn đến thư mục chứa file âm thanh
    output_csv : str
        Đường dẫn đến file CSV đầu ra
    db_path : str
        Đường dẫn đến file database
    """
    all_features = []
    
    # Lấy danh sách file âm thanh
    audio_files = [f for f in os.listdir(directory_path) 
                  if f.endswith(('.wav', '.mp3', '.ogg', '.flac'))]
    
    for file in tqdm(audio_files, desc="Đang xử lý file âm thanh"):
        file_path = os.path.join(directory_path, file)
        
        # Trích xuất đặc trưng
        try:
            features = extract_features(file_path)
            all_features.extend(features)
        except Exception as e:
            print(f"Lỗi khi xử lý file {file}: {e}")
    
    # Lưu tất cả đặc trưng vào CSV
    if all_features:
        df = save_features_to_csv(all_features, output_csv)
        
        # Cập nhật database
        update_database(df, db_path)
    else:
        print("Không có đặc trưng nào được trích xuất.")

if __name__ == "__main__":
    # Cài đặt đường dẫn và các tham số
    audio_directory = "audio_files"  # Thay đổi thành thư mục chứa file âm thanh
    output_csv = "features.csv"      # File CSV đầu ra
    db_path = "audio_features.db"    # File database
    
    # Xử lý thư mục âm thanh
    process_audio_directory(audio_directory, output_csv, db_path)
    
    print("Hoàn thành quá trình trích xuất đặc trưng và cập nhật database.") 