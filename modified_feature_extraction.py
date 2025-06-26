import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
import os
import joblib
import io

def extract_features(audio_path):
    """
    Trích xuất 29 vector đặc trưng theo bars với overlap 50% + 1 vector tổng + BPM cố định
    
    Parameters:
    -----------
    audio_path : str
        Đường dẫn đến file âm thanh
        
    Returns:
    --------
    features : dict
        Dictionary chứa 29 vector đặc trưng + 1 vector tổng + BPM
    """
    try:
        # 1. Đọc file âm thanh
        sr, audio = wav.read(audio_path)
        
        # Đảm bảo dữ liệu là dạng mono và float32
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # Convert to mono
        audio = audio.astype(np.float32)
        
        # Chuẩn hóa âm thanh
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # 2. Thiết lập tham số theo yêu cầu
        BPM = 120  # Cố định 120 BPM cho tất cả
        beats_per_bar = 4  # 4/4 time signature
        bar_duration = 60 / BPM * beats_per_bar  # 2 giây cho mỗi bar
        
        # Tính frame size và hop size
        bar_samples = int(bar_duration * sr)  # số sample trong 1 bar (2 giây)
        hop_samples = bar_samples // 2  # overlap 50%
        
        # 3. Chia âm thanh thành các bars với overlap 50%
        audio_length = len(audio)
        num_bars = int((audio_length - bar_samples) / hop_samples) + 1
        
        # Đảm bảo có ít nhất 29 bars để tạo 29 vector
        if num_bars < 29:
            # Nếu file quá ngắn, lặp lại âm thanh để đủ 29 bars
            repeat_times = int(np.ceil((29 * hop_samples + bar_samples) / audio_length))
            audio = np.tile(audio, repeat_times)
            audio_length = len(audio)
            num_bars = int((audio_length - bar_samples) / hop_samples) + 1
        
        # Lấy chính xác 29 bars đầu tiên
        num_bars = min(num_bars, 29)
        
        # 4. Trích xuất đặc trưng cho từng bar
        bar_features = []
        all_frame_features = []  # Lưu tất cả features để tính vector tổng
        
        for bar_idx in range(num_bars):
            # Lấy dữ liệu của bar hiện tại
            start_sample = bar_idx * hop_samples
            end_sample = start_sample + bar_samples
            bar_audio = audio[start_sample:min(end_sample, audio_length)]
            
            # Padding nếu bar không đủ dài
            if len(bar_audio) < bar_samples:
                bar_audio = np.pad(bar_audio, (0, bar_samples - len(bar_audio)))
            
            # 5. Tính đặc trưng cho bar này
            # Đặc trưng 1: RMS Energy
            rms = np.sqrt(np.mean(bar_audio ** 2))
            
            # Đặc trưng 2: Zero-Crossing Rate  
            zcr = np.sum(np.abs(np.diff(np.signbit(bar_audio)))) / (2 * len(bar_audio))
            
            # Đặc trưng 3-5: Spectral features
            # Tính FFT
            window = np.hamming(len(bar_audio))
            windowed_audio = bar_audio * window
            fft_spectrum = np.abs(np.fft.rfft(windowed_audio))
            frequencies = np.fft.rfftfreq(len(bar_audio), 1/sr)
            
            # Tránh chia cho 0
            total_spectrum = np.sum(fft_spectrum)
            if total_spectrum == 0:
                spectral_centroid = 0
                spectral_bandwidth = 0
                spectral_rolloff = 0
            else:
                # Spectral Centroid
                spectral_centroid = np.sum(frequencies * fft_spectrum) / total_spectrum
                
                # Spectral Bandwidth
                spectral_bandwidth = np.sqrt(np.sum(((frequencies - spectral_centroid) ** 2) * fft_spectrum) / total_spectrum)
                
                # Spectral Rolloff (85%)
                cumsum_spectrum = np.cumsum(fft_spectrum)
                rolloff_threshold = 0.85 * total_spectrum
                rolloff_idx = np.where(cumsum_spectrum >= rolloff_threshold)[0]
                if len(rolloff_idx) > 0:
                    spectral_rolloff = frequencies[rolloff_idx[0]]
                else:
                    spectral_rolloff = 0
            
            # Tạo vector đặc trưng cho bar này
            bar_feature_vector = np.array([
                rms,
                zcr, 
                spectral_centroid,
                spectral_bandwidth,
                spectral_rolloff
            ])
            
            bar_features.append(bar_feature_vector)
            all_frame_features.append(bar_feature_vector)
        
        # 6. Tạo vector tổng (trung bình của 29 bars)
        all_features_array = np.array(all_frame_features)
        summary_vector = np.mean(all_features_array, axis=0)
        
        # 7. Chuẩn bị kết quả
        features = {
            'bpm': BPM,
            'num_bars': num_bars,
            'bar_duration': bar_duration,
            'bar_features': bar_features,  # List của 29 vector đặc trưng
            'summary_vector': summary_vector,  # 1 vector tổng
            
            # Compatibility với code cũ
            'rms_energy': summary_vector[0],
            'zcr': summary_vector[1], 
            'spectral_centroid': summary_vector[2],
            'spectral_bandwidth': summary_vector[3],
            'spectral_rolloff': summary_vector[4],
            'tempo': BPM
        }
        
        # 8. Tạo feature_vector chính để so sánh
        # Kết hợp: 29 bars * 5 features + 1 summary vector * 5 features + 1 BPM = 151 features
        feature_vector = []
        
        # Thêm 29 vector bars
        for bar_feat in bar_features:
            feature_vector.extend(bar_feat)
            
        # Thêm 1 vector tổng
        feature_vector.extend(summary_vector)
        
        # Thêm BPM
        feature_vector.append(BPM)
        
        features['feature_vector'] = np.array(feature_vector)
        
        return features
    
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None

def extract_and_save_features(audio_dir, db_path):
    """
    Trích xuất đặc trưng từ tất cả files âm thanh trong thư mục và lưu vào cơ sở dữ liệu
    
    Parameters:
    -----------
    audio_dir : str
        Đường dẫn đến thư mục chứa các file âm thanh
    db_path : str
        Đường dẫn đến file cơ sở dữ liệu SQLite
    """
    import sqlite3
    from tqdm import tqdm
    import mutagen
    
    # Kết nối đến cơ sở dữ liệu
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Tạo bảng nếu chưa tồn tại với cấu trúc hiện tại
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS songs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        title TEXT,
        genre TEXT,
        features BLOB NOT NULL,
        feature_count INTEGER,
        feature_version TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    
    # Lấy danh sách các file âm thanh
    audio_files = []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(('.mp3', '.wav', '.ogg')):
                audio_files.append(os.path.join(root, file))
    
    # Trích xuất đặc trưng và lưu vào cơ sở dữ liệu
    for audio_path in tqdm(audio_files, desc="Extracting features"):
        features = extract_features(audio_path)
        if features is not None:
            # Chuyển đổi đặc trưng thành dạng binary để lưu vào SQLite
            buffer = io.BytesIO()
            joblib.dump(features, buffer)
            features_blob = buffer.getvalue()
            
            # Lấy thông tin cơ bản về file
            filename = os.path.basename(audio_path)
            
            # Trích xuất thông tin từ filename
            title = os.path.splitext(filename)[0]
            # Lấy genre từ tên file (format: genre.number.wav)
            genre = filename.split('.')[0] if '.' in filename else 'unknown'
            
            # Kiểm tra xem file đã tồn tại trong cơ sở dữ liệu chưa
            cursor.execute("SELECT id FROM songs WHERE filename = ?", (filename,))
            existing = cursor.fetchone()
            
            if existing:
                # Cập nhật nếu đã tồn tại
                cursor.execute(
                    "UPDATE songs SET features = ?, title = ?, genre = ?, feature_count = ?, feature_version = ? WHERE filename = ?",
                    (features_blob, title, genre, len(features['feature_vector']), "bars_v1", filename)
                )
            else:
                # Thêm mới nếu chưa tồn tại
                cursor.execute(
                    "INSERT INTO songs (filename, title, genre, features, feature_count, feature_version) VALUES (?, ?, ?, ?, ?, ?)",
                    (filename, title, genre, features_blob, len(features['feature_vector']), "bars_v1")
                )
            
            conn.commit()
    
    conn.close()
    print(f"Extracted features from {len(audio_files)} audio files and saved to database.")

def retrieve_features(db_path, song_id=None, filename=None):
    """
    Lấy đặc trưng âm thanh từ cơ sở dữ liệu dựa trên ID hoặc tên file
    
    Parameters:
    -----------
    db_path : str
        Đường dẫn đến file cơ sở dữ liệu SQLite
    song_id : int, optional
        ID của bài hát trong cơ sở dữ liệu
    filename : str, optional
        Tên file của bài hát
        
    Returns:
    --------
    song_info : dict
        Thông tin về bài hát và đặc trưng âm thanh
    """
    import sqlite3
    
    if song_id is None and filename is None:
        raise ValueError("Phải cung cấp song_id hoặc filename")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    if song_id is not None:
        cursor.execute("SELECT id, filename, title, artist, album, features FROM songs WHERE id = ?", (song_id,))
    else:
        cursor.execute("SELECT id, filename, title, artist, album, features FROM songs WHERE filename = ?", (filename,))
    
    result = cursor.fetchone()
    conn.close()
    
    if result is None:
        return None
    
    song_id, filename, title, artist, album, features_blob = result
    
    # Giải nén dữ liệu đặc trưng
    features = joblib.load(io.BytesIO(features_blob))
    
    return {
        'id': song_id,
        'filename': filename,
        'title': title,
        'artist': artist,
        'album': album,
        'features': features
    }

def get_feature_vectors(db_path):
    """
    Lấy danh sách vector đặc trưng của tất cả các bài hát trong cơ sở dữ liệu
    
    Parameters:
    -----------
    db_path : str
        Đường dẫn đến file cơ sở dữ liệu SQLite
        
    Returns:
    --------
    song_features : list
        Danh sách các dict chứa ID bài hát và vector đặc trưng
    """
    import sqlite3
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, filename, title, genre, features FROM songs")
    results = cursor.fetchall()
    conn.close()
    
    song_features = []
    
    for song_id, filename, title, genre, features_blob in results:
        try:
            # Giải nén dữ liệu đặc trưng
            features = joblib.load(io.BytesIO(features_blob))
            
            # Xử lý cả format cũ (numpy array) và mới (dict)
            if isinstance(features, dict):
                # Format mới - features là dict
                feature_vector = features['feature_vector']
            else:
                # Format cũ - features là numpy array
                feature_vector = features
            
            song_features.append({
                'id': song_id,
                'filename': filename,
                'title': title,
                'genre': genre,
                'feature_vector': feature_vector
            })
        except Exception as e:
            print(f"Lỗi khi đọc features cho {filename}: {e}")
            continue
    
    return song_features 