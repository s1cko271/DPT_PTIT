import numpy as np
import scipy.io.wavfile as wavfile
from scipy import signal
from scipy.fft import fft
import os
import joblib
import io
import csv
import matplotlib.pyplot as plt

def lowpass_filter(signal, sr, cutoff_freq):
    """
    Áp dụng bộ lọc thông thấp cho tín hiệu
    
    Tham số:
        signal: tín hiệu đầu vào
        sr: tần số lấy mẫu
        cutoff_freq: tần số cắt
        
    Trả về:
        filtered_signal: tín hiệu đã lọc
    """
    nyquist = 0.5 * sr
    normal_cutoff = cutoff_freq / nyquist
    # Thiết kế bộ lọc Butterworth bậc 6
    b, a = signal.butter(6, normal_cutoff, btype='lowpass')
    filtered_signal = signal.filtfilt(b, a, signal)
    return filtered_signal

def frame_signal(signal, frame_size, hop_size):
    """
    Chia tín hiệu thành các khung chồng lấp
    """
    num_frames = 1 + (len(signal) - frame_size) // hop_size
    frames = np.zeros((num_frames, frame_size))
    for i in range(num_frames):
        start = i * hop_size
        frames[i] = signal[start:start + frame_size]
    return frames

def compute_energy(frames):
    """
    Tính năng lượng cho mỗi khung tín hiệu
    """
    return np.sum(frames**2, axis=1)

def normalize_signal(signal):
    """
    Chuẩn hóa tín hiệu về khoảng [0, 1]
    """
    min_val = np.min(signal)
    max_val = np.max(signal)
    if max_val == min_val:
        return np.zeros_like(signal)
    return (signal - min_val) / (max_val - min_val)

def autocorrelation(signal):
    """
    Tính tự tương quan của tín hiệu
    """
    result = np.correlate(signal, signal, mode='full')
    return result[result.size // 2:]

def tempo_estimate(wav_file_path, plot=False, min_bpm=40, max_bpm=220):
    """
    Ước lượng tempo (BPM) từ file WAV
    
    Tham số:
        wav_file_path: đường dẫn đến file WAV
        plot: True nếu muốn hiển thị biểu đồ kết quả
        min_bpm: giới hạn dưới của BPM
        max_bpm: giới hạn trên của BPM
        
    Trả về:
        bpm: tempo ước lượng (BPM)
    """
    # Đọc file âm thanh
    sr, audio_signal = wavfile.read(wav_file_path)
    
    # Chuẩn hóa tín hiệu về khoảng [-1, 1]
    if audio_signal.dtype == np.int16:
        audio_signal = audio_signal.astype(np.float32) / 32768.0
        
    # Áp dụng bộ lọc dải thông từ 20-150Hz để tập trung vào vùng tần số của kick/bass
    filtered_signal = lowpass_filter(audio_signal, sr, 120)
    
    # Chia tín hiệu thành các khung
    frame_size = 512  # khoảng 23ms với sr=22050
    hop_size = 256    # 50% overlap
    frames = frame_signal(filtered_signal, frame_size, hop_size)
    
    # Tính năng lượng của từng khung
    energy = compute_energy(frames)
    
    # Làm mịn đường năng lượng với bộ lọc thông thấp
    energy_smooth = lowpass_filter(energy, sr / hop_size, 5)
    
    # Chuẩn hóa năng lượng
    energy_smooth = normalize_signal(energy_smooth)
    
    # Tính autocorrelation
    corr = autocorrelation(energy_smooth)
    
    # Tìm các đỉnh trong hàm tự tương quan
    # Bỏ qua các đỉnh ở đầu (tương ứng với tempo quá cao)
    min_lag = int(sr / hop_size * 60 / max_bpm)  # Giới hạn 220 BPM
    max_lag = int(sr / hop_size * 60 / min_bpm)   # Giới hạn 40 BPM
    
    # Giới hạn phạm vi tìm kiếm
    corr_trim = corr[min_lag:max_lag]
    
    # Tìm đỉnh cao nhất
    peaks, _ = signal.find_peaks(corr_trim, height=0)
    
    if len(peaks) == 0:
        return 60.0  # Trả về giá trị mặc định nếu không tìm thấy đỉnh
    
    # Lấy đỉnh cao nhất
    peak_heights = corr_trim[peaks]
    strongest_peak_idx = peaks[np.argmax(peak_heights)]
    
    # Tính lag thực (vị trí đỉnh + vị trí bắt đầu tìm kiếm)
    lag_samples = strongest_peak_idx + min_lag
    
    # Tính thời gian tương ứng với lag này
    lag_time = lag_samples * hop_size / sr
    
    # Tính BPM
    bpm = 60.0 / lag_time
    
    # Vẽ biểu đồ nếu cần
    if plot:
        plt.figure(figsize=(12, 8))
        
        # Plot tín hiệu gốc
        plt.subplot(4, 1, 1)
        t = np.arange(len(audio_signal)) / sr
        plt.plot(t, audio_signal)
        plt.title("Tín hiệu gốc")
        plt.xlabel("Thời gian (s)")
        
        # Plot tín hiệu đã lọc
        plt.subplot(4, 1, 2)
        t_filter = np.arange(len(filtered_signal)) / sr
        plt.plot(t_filter, filtered_signal)
        plt.title("Tín hiệu sau khi lọc dải thông thấp 150Hz")
        plt.xlabel("Thời gian (s)")
        
        # Plot năng lượng
        plt.subplot(4, 1, 3)
        t_energy = np.arange(len(energy_smooth)) * hop_size / sr
        plt.plot(t_energy, energy_smooth)
        plt.title("Đường bao năng lượng")
        plt.xlabel("Thời gian (s)")
        
        # Plot autocorrelation và đánh dấu đỉnh
        plt.subplot(4, 1, 4)
        t_corr = np.arange(len(corr)) * hop_size / sr
        plt.plot(t_corr[:max_lag], corr[:max_lag])
        plt.axvline(x=lag_time, color='r', linestyle='--')
        plt.text(lag_time + 0.1, 0.5, f"{bpm:.1f} BPM", color='r')
        plt.title("Hàm tự tương quan")
        plt.xlabel("Độ trễ (s)")
        
        plt.tight_layout()
        plt.show()
    
    return bpm

def extract_features(audio_path):
    """
    Trích xuất đặc trưng âm thanh từ file WAV
    
    Parameters:
    -----------
    audio_path : str
        Đường dẫn đến file âm thanh
        
    Returns:
    --------
    features : dict
        Dictionary chứa các đặc trưng âm thanh
    """
    try:
        # Đọc file WAV
        sampling_rate, audio_data = wavfile.read(audio_path)
        
        # Chuyển đổi sang kiểu float để xử lý
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0  # Chuẩn hóa về [-1, 1]
        
        # Chuyển về mono nếu là stereo
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        # Ước lượng BPM của bài hát
        try:
            bpm = tempo_estimate(audio_path)
        except:
            # Nếu không ước lượng được thì dùng giá trị mặc định
            bpm = 120
        
        # Tính toán độ dài của frame dựa trên BPM
        # 1 bar = 4 beats
        bar_duration = 60 / bpm * 4  # tính bằng giây
        frame_length = int(bar_duration * sampling_rate)
        hop_size = frame_length // 2  # 50% overlap
        
        # Chuẩn bị đặc trưng
        features_list = []
        
        # Xử lý từng frame
        for frame_index, i in enumerate(range(0, len(audio_data) - frame_length + 1, hop_size)):
            frame = audio_data[i:i + frame_length]
            start_time = i / sampling_rate
            
            # RMS energy
            rms = np.sqrt(np.mean(frame ** 2))
            
            # Average Energy
            avg_energy = np.mean(frame ** 2)
            
            # Zero-Crossing Rate
            zcr = np.sum(np.abs(np.diff(np.signbit(frame)))) / len(frame)
            
            # Tính toán phổ (magnitude spectrum)
            spectrum = np.abs(fft(frame))
            spectrum = spectrum[:frame_length // 2 + 1]  # Lấy nửa đầu (phổ dương)
            
            # Tạo vector tần số
            frequencies = np.linspace(0, sampling_rate // 2, len(spectrum))
            
            # Tính tổng phổ (để chuẩn hóa)
            total_spectrum = np.sum(spectrum)
            if total_spectrum == 0:
                # Tránh chia cho 0
                centroid = 0
                bandwidth = 0
                rolloff = 0
            else:
                # Spectral Centroid
                centroid = np.sum(frequencies * spectrum) / total_spectrum
                
                # Spectral Bandwidth (độ lệch chuẩn xung quanh centroid)
                bandwidth = np.sqrt(np.sum(((frequencies - centroid) ** 2) * spectrum) / total_spectrum)
                
                # Spectral Rolloff (85%)
                threshold = 0.85 * total_spectrum
                cumsum_spectrum = np.cumsum(spectrum)
                rolloff_idx = np.where(cumsum_spectrum >= threshold)[0][0]
                rolloff = frequencies[rolloff_idx]
            
            # Lưu các đặc trưng cho frame hiện tại
            features_list.append([
                frame_index,        # Chỉ số frame
                start_time,         # Thời điểm bắt đầu (giây)
                rms,                # RMS energy
                zcr,                # Zero-Crossing Rate
                centroid,           # Spectral Centroid
                bandwidth,          # Spectral Bandwidth
                rolloff,            # Spectral Rolloff
                avg_energy          # Average Energy
            ])
        
        # Tính trung bình các đặc trưng của tất cả các frame
        features_array = np.array(features_list)
        avg_features = np.mean(features_array[:, 2:], axis=0)  # Bỏ qua frame_index và start_time
        
        # Tạo dictionary đặc trưng
        features = {
            'rms_energy': avg_features[0],
            'zcr': avg_features[1],
            'spectral_centroid': avg_features[2],
            'spectral_bandwidth': avg_features[3],
            'spectral_rolloff': avg_features[4],
            'avg_energy': avg_features[5],
            'bpm': bpm
        }
        
        # Thêm vector đặc trưng để dễ dàng tính khoảng cách
        features['feature_vector'] = np.array([
            features['rms_energy'],
            features['zcr'],
            features['spectral_centroid'],
            features['spectral_bandwidth'],
            features['spectral_rolloff'],
            features['avg_energy'],
            features['bpm']
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
    
    # Giải nén đặc trưng từ binary blob
    features = joblib.load(io.BytesIO(result[5]))
    
    return {
        'id': result[0],
        'filename': result[1],
        'title': result[2],
        'artist': result[3],
        'album': result[4],
        'features': features
    }

def search_similar_songs(db_path, query_features, num_results=3):
    """
    Tìm kiếm các bài hát tương tự dựa trên đặc trưng âm thanh sử dụng Cosine Similarity
    
    Parameters:
    -----------
    db_path : str
        Đường dẫn đến file cơ sở dữ liệu SQLite
    query_features : dict
        Dictionary chứa các đặc trưng âm thanh từ bài hát truy vấn
    num_results : int, optional
        Số lượng kết quả cần trả về (mặc định: 3)
        
    Returns:
    --------
    similar_songs : list
        Danh sách các bài hát tương tự, sắp xếp theo độ tương đồng
    """
    import sqlite3
    from sklearn.metrics.pairwise import cosine_similarity
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Lấy tất cả bài hát từ database
    cursor.execute("SELECT id, filename, title, artist, album, features FROM songs")
    all_songs = cursor.fetchall()
    conn.close()
    
    query_vector = query_features['feature_vector']
    
    # Tính toán độ tương tự với tất cả bài hát trong database
    similarities = []
    for song in all_songs:
        song_id, filename, title, artist, album, features_blob = song
        
        features = joblib.load(io.BytesIO(features_blob))
        song_vector = features['feature_vector']
        
        # Cosine similarity (càng cao càng giống nhau)
        similarity = cosine_similarity([query_vector], [song_vector])[0][0]
        
        similarities.append({
            'id': song_id,
            'filename': filename,
            'title': title,
            'artist': artist,
            'album': album,
            'similarity': similarity
        })
    
    # Sắp xếp theo độ tương tự (giảm dần)
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Trả về top N kết quả
    return similarities[:num_results]

def get_feature_vectors(db_path):
    """
    Lấy tất cả feature vectors từ cơ sở dữ liệu để phân tích
    
    Parameters:
    -----------
    db_path : str
        Đường dẫn đến file cơ sở dữ liệu SQLite
        
    Returns:
    --------
    feature_vectors : dict
        Dictionary với key là ID bài hát và value là feature vector
    """
    import sqlite3
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, features FROM songs")
    all_songs = cursor.fetchall()
    conn.close()
    
    feature_vectors = {}
    for song_id, features_blob in all_songs:
        features = joblib.load(io.BytesIO(features_blob))
        feature_vectors[song_id] = features['feature_vector']
    
    return feature_vectors

def normalize_feature_vectors(feature_vectors):
    """
    Chuẩn hóa các feature vector để so sánh tốt hơn
    
    Parameters:
    -----------
    feature_vectors : dict
        Dictionary với key là ID bài hát và value là feature vector
        
    Returns:
    --------
    normalized_vectors : dict
        Dictionary với key là ID bài hát và value là feature vector đã chuẩn hóa
    """
    # Chuyển đổi dictionary thành numpy array
    song_ids = list(feature_vectors.keys())
    vectors = np.array([feature_vectors[song_id] for song_id in song_ids])
    
    # Tính mean và std cho từng feature
    means = np.mean(vectors, axis=0)
    stds = np.std(vectors, axis=0)
    
    # Thay thế các std = 0 bằng 1 để tránh chia cho 0
    stds[stds == 0] = 1
    
    # Chuẩn hóa
    normalized = (vectors - means) / stds
    
    # Chuyển đổi trở lại thành dictionary
    normalized_vectors = {}
    for i, song_id in enumerate(song_ids):
        normalized_vectors[song_id] = normalized[i]
    
    return normalized_vectors 