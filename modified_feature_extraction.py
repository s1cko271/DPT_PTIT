import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
import os
import joblib
import io

def extract_features(audio_path, frame_size=2048, hop_size=512):
    """
    Trích xuất đặc trưng đơn giản từ file âm thanh, không phụ thuộc thư viện phức tạp
    
    Parameters:
    -----------
    audio_path : str
        Đường dẫn đến file âm thanh
    frame_size : int
        Kích thước khung (mặc định: 2048 mẫu)
    hop_size : int
        Bước nhảy giữa các khung (mặc định: 512 mẫu)
        
    Returns:
    --------
    features : dict
        Dictionary chứa các đặc trưng âm thanh đơn giản
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
        
        # 2. Chia âm thanh thành các khung
        num_frames = int((len(audio) - frame_size) / hop_size) + 1
        frames = np.zeros((num_frames, frame_size))
        
        for i in range(num_frames):
            start = i * hop_size
            end = min(start + frame_size, len(audio))
            frame = audio[start:end]
            if len(frame) < frame_size:
                # Padding nếu frame không đủ dài
                frame = np.pad(frame, (0, frame_size - len(frame)))
            frames[i] = frame
        
        # 3. Tính các đặc trưng cho từng khung
        
        # Đặc trưng 1: RMS Energy (độ to của âm thanh)
        rms_energy = np.sqrt(np.mean(frames**2, axis=1))
        
        # Đặc trưng 2: Zero-Crossing Rate (tần suất tín hiệu đổi dấu)
        zcr = np.sum(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1) / (2 * frame_size)
        
        # Đặc trưng 3-5: Spectral features
        # Tính FFT cho từng khung
        window = np.hamming(frame_size)
        windowed_frames = frames * window
        fft_frames = np.abs(np.fft.rfft(windowed_frames, axis=1))
        frequencies = np.fft.rfftfreq(frame_size, 1/sr)
        
        # Đặc trưng 3: Spectral Centroid (trung tâm phổ - "độ sáng")
        spectral_centroid = np.sum(frequencies.reshape(1, -1) * fft_frames, axis=1) / (np.sum(fft_frames, axis=1) + 1e-8)
        
        # Đặc trưng 4: Spectral Bandwidth (độ rộng phổ)
        spectral_bandwidth = np.sqrt(np.sum(((frequencies.reshape(1, -1) - spectral_centroid.reshape(-1, 1))**2) * fft_frames, axis=1) / (np.sum(fft_frames, axis=1) + 1e-8))
        
        # Đặc trưng 5: Spectral Rolloff (tần số mà 85% năng lượng nằm bên trái)
        cumsum = np.cumsum(fft_frames, axis=1)
        rolloff_point = 0.85 * np.sum(fft_frames, axis=1).reshape(-1, 1)
        spectral_rolloff = np.zeros(num_frames)
        
        for i in range(num_frames):
            rolloff_idx = np.where(cumsum[i] >= rolloff_point[i])[0]
            if len(rolloff_idx) > 0:
                spectral_rolloff[i] = frequencies[rolloff_idx[0]]
            else:
                spectral_rolloff[i] = 0
        
        # Đặc trưng 6: Tempo (BPM) bằng autocorrelation
        # Chuẩn hóa RMS energy
        rms_norm = rms_energy - np.mean(rms_energy)
        rms_norm = rms_norm / (np.std(rms_norm) + 1e-8)
        
        # Tính autocorrelation
        ac = np.correlate(rms_norm, rms_norm, mode='full')
        ac = ac[len(ac)//2:]  # Chỉ lấy phần dương
        
        # Tìm peak đầu tiên trong vùng có ý nghĩa (60-200 BPM)
        min_lag = int(sr / hop_size * 60 / 200)  # 200 BPM
        max_lag = int(sr / hop_size * 60 / 60)   # 60 BPM
        
        # Xử lý trường hợp file quá ngắn
        if min_lag < len(ac) and max_lag < len(ac):
            ac_crop = ac[min_lag:max_lag]
            lag = np.argmax(ac_crop) + min_lag
            tempo = 60 * sr / (lag * hop_size)
        else:
            tempo = 120  # Giá trị mặc định nếu không tính được
        
        # Tổng hợp các đặc trưng - chỉ giữ 6 đặc trưng đơn giản
        features = {
            'rms_energy': np.mean(rms_energy),
            'zcr': np.mean(zcr),
            'spectral_centroid': np.mean(spectral_centroid),
            'spectral_bandwidth': np.mean(spectral_bandwidth),
            'spectral_rolloff': np.mean(spectral_rolloff),
            'tempo': tempo
        }
        
        # Tạo thêm vector đặc trưng (dễ dàng cho tính toán khoảng cách)
        features['feature_vector'] = np.array([
            features['rms_energy'],
            features['zcr'],
            features['spectral_centroid'],
            features['spectral_bandwidth'],
            features['spectral_rolloff'],
            features['tempo']
        ])
        
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
    
    # Tạo bảng nếu chưa tồn tại 
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS songs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        title TEXT,
        artist TEXT,
        album TEXT,
        features BLOB NOT NULL,
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
            
            # Trích xuất metadata từ file nhạc
            try:
                audio_metadata = mutagen.File(audio_path, easy=True)
                title = audio_metadata.get('title', [os.path.splitext(filename)[0]])[0]
                artist = audio_metadata.get('artist', ['Unknown'])[0]
                album = audio_metadata.get('album', ['Unknown'])[0]
            except:
                # Nếu không đọc được metadata, sử dụng thông tin mặc định
                title = os.path.splitext(filename)[0]
                artist = 'Unknown'
                album = 'Unknown'
            
            # Kiểm tra xem file đã tồn tại trong cơ sở dữ liệu chưa
            cursor.execute("SELECT id FROM songs WHERE filename = ?", (filename,))
            existing = cursor.fetchone()
            
            if existing:
                # Cập nhật nếu đã tồn tại
                cursor.execute(
                    "UPDATE songs SET features = ?, title = ?, artist = ?, album = ? WHERE filename = ?",
                    (features_blob, title, artist, album, filename)
                )
            else:
                # Thêm mới nếu chưa tồn tại
                cursor.execute(
                    "INSERT INTO songs (filename, title, artist, album, features) VALUES (?, ?, ?, ?, ?)",
                    (filename, title, artist, album, features_blob)
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
    
    cursor.execute("SELECT id, filename, title, artist, features FROM songs")
    results = cursor.fetchall()
    conn.close()
    
    song_features = []
    
    for song_id, filename, title, artist, features_blob in results:
        # Giải nén dữ liệu đặc trưng
        features = joblib.load(io.BytesIO(features_blob))
        
        song_features.append({
            'id': song_id,
            'filename': filename,
            'title': title,
            'artist': artist,
            'feature_vector': features['feature_vector']
        })
    
    return song_features 