import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
import os
import joblib
import io

def extract_features(audio_path, frame_size=2048, hop_size=512):
    """
    Trích xuất đặc trưng đơn giản từ file âm thanh
    
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
        audio = audio / np.max(np.abs(audio))
        
        # 2. Tính RMS Energy (độ to/nhỏ của âm thanh)
        def rms(x):
            return np.sqrt(np.mean(x**2))
        
        rms_energy = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i+frame_size]
            rms_energy.append(rms(frame))
        
        # 3. Tính Zero Crossing Rate (tần suất tín hiệu đi qua 0)
        zcr = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i+frame_size]
            zcr.append(np.sum(np.abs(np.diff(np.signbit(frame)))) / (2 * len(frame)))
        
        # 4. Tính Spectral Centroid (trọng tâm của spectrum)
        spectral_centroid = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i+frame_size]
            magnitude_spectrum = np.abs(np.fft.rfft(frame * np.hamming(len(frame))))
            freqs = np.fft.rfftfreq(frame_size, d=1/sr)
            
            # Tránh chia cho 0
            if np.sum(magnitude_spectrum) > 0:
                spectral_centroid.append(np.sum(freqs * magnitude_spectrum) / np.sum(magnitude_spectrum))
            else:
                spectral_centroid.append(0)
        
        # 5. Tính Spectral Bandwidth (độ rộng của spectrum)
        spectral_bandwidth = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i+frame_size]
            magnitude_spectrum = np.abs(np.fft.rfft(frame * np.hamming(len(frame))))
            freqs = np.fft.rfftfreq(frame_size, d=1/sr)
            
            # Tính spectral centroid cho frame hiện tại
            if np.sum(magnitude_spectrum) > 0:
                centroid = np.sum(freqs * magnitude_spectrum) / np.sum(magnitude_spectrum)
                # Tính bandwidth
                bandwidth = np.sqrt(np.sum(((freqs - centroid)**2) * magnitude_spectrum) / np.sum(magnitude_spectrum))
                spectral_bandwidth.append(bandwidth)
            else:
                spectral_bandwidth.append(0)
        
        # 6. Tính Spectral Rolloff (tần số mà phổ tích lũy đạt được ngưỡng)
        rolloff_threshold = 0.85
        spectral_rolloff = []
        
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i+frame_size]
            magnitude_spectrum = np.abs(np.fft.rfft(frame * np.hamming(len(frame))))
            freqs = np.fft.rfftfreq(frame_size, d=1/sr)
            
            # Tính tổng tích lũy
            cumsum = np.cumsum(magnitude_spectrum)
            if cumsum[-1] > 0:
                normalized_cumsum = cumsum / cumsum[-1]
                # Tìm tần số rolloff
                rolloff_index = np.argmax(normalized_cumsum >= rolloff_threshold)
                if rolloff_index > 0:
                    spectral_rolloff.append(freqs[rolloff_index])
                else:
                    spectral_rolloff.append(0)
            else:
                spectral_rolloff.append(0)
        
        # 7. Tính Tempo bằng phương pháp tự tương quan
        # Phân tích trong khoảng BPM 60-180
        win_size = int(sr * 60 / 60)  # Window size for 60 BPM
        hop_length = int(win_size / 2)
        
        # Tính envelope
        envelope = np.abs(signal.hilbert(audio))
        
        # Tính autocorrelation
        corr = signal.correlate(envelope, envelope, mode='full')
        corr = corr[len(corr)//2:]
        
        # Tìm các peak trong autocorrelation
        min_bpm, max_bpm = 60, 180
        min_lag = sr * 60 // max_bpm
        max_lag = sr * 60 // min_bpm
        
        if max_lag < len(corr):
            # Cắt autocorrelation trong khoảng BPM mong muốn
            corr = corr[min_lag:max_lag]
            peaks, _ = signal.find_peaks(corr)
            
            if len(peaks) > 0:
                # Tìm peak lớn nhất
                peak_idx = min_lag + peaks[np.argmax(corr[peaks])]
                tempo = sr * 60 / peak_idx
            else:
                tempo = 120  # Default tempo
        else:
            tempo = 120  # Default tempo
        
        # Đóng gói đặc trưng
        features = {
            'rms_energy': np.mean(rms_energy),
            'zcr': np.mean(zcr),
            'spectral_centroid': np.mean(spectral_centroid),
            'spectral_bandwidth': np.mean(spectral_bandwidth),
            'spectral_rolloff': np.mean(spectral_rolloff),
            'tempo': tempo
        }
        
        # Tạo vector đặc trưng
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

def save_features_to_db(features, db_path='audio_features.db'):
    """
    Lưu đặc trưng vào cơ sở dữ liệu
    
    Parameters:
    -----------
    features : dict
        Dictionary chứa đặc trưng âm thanh
    db_path : str
        Đường dẫn đến file database
    """
    import sqlite3
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Tạo bảng nếu chưa tồn tại
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        rms_energy REAL,
        zcr REAL,
        spectral_centroid REAL,
        spectral_bandwidth REAL,
        spectral_rolloff REAL,
        tempo REAL,
        feature_vector BLOB
    )
    ''')
    
    # Chuyển vector feature thành blob
    buffer = io.BytesIO()
    joblib.dump(features['feature_vector'], buffer)
    feature_vector_blob = buffer.getvalue()
    
    # Lưu đặc trưng
    cursor.execute('''
    INSERT INTO features (rms_energy, zcr, spectral_centroid, spectral_bandwidth, spectral_rolloff, tempo, feature_vector) 
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        features['rms_energy'],
        features['zcr'],
        features['spectral_centroid'],
        features['spectral_bandwidth'],
        features['spectral_rolloff'],
        features['tempo'],
        feature_vector_blob
    ))
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    # Test đoạn code với một file âm thanh mẫu
    audio_path = "audio_files/sample.wav"
    if os.path.exists(audio_path):
        features = extract_features(audio_path)
        print("Extracted features:", features)
        save_features_to_db(features)
        print("Features saved to database") 