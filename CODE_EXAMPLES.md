# 💻 CODE EXAMPLES VÀ LỆNH QUAN TRỌNG

## 🚀 LỆNH CHẠY DEMO
```bash
# Khởi động demo
python -m streamlit run streamlit_demo.py

# Test pipeline từ command line
python audio_processing_pipeline.py --audio songs/blues.00000.wav --method cosine

# Visualize results
python visualize_results.py --audio songs/jazz.00000.wav
```

## 🎵 CORE FEATURE EXTRACTION CODE

### 1. RMS Energy Calculation
```python
def calculate_rms(audio_signal):
    """Tính RMS Energy - đo cường độ âm thanh"""
    rms = np.sqrt(np.mean(audio_signal ** 2))
    return rms

# Ví dụ sử dụng
audio = [0.1, -0.5, 0.3, -0.2, 0.4]
rms_value = calculate_rms(audio)
print(f"RMS Energy: {rms_value}")
```

### 2. Zero-Crossing Rate
```python
def calculate_zcr(audio_signal):
    """Tính Zero-Crossing Rate - tần số đổi dấu"""
    # Đếm số lần tín hiệu đổi dấu
    sign_changes = np.abs(np.diff(np.signbit(audio_signal)))
    zcr = np.sum(sign_changes) / (2 * len(audio_signal))
    return zcr

# Ví dụ
audio = [1, -1, 1, -1, 1]  # High ZCR
zcr_value = calculate_zcr(audio)
print(f"ZCR: {zcr_value}")
```

### 3. Spectral Features
```python
def calculate_spectral_features(audio_signal, sample_rate):
    """Tính các đặc trưng phổ tần số"""
    # FFT để chuyển sang domain tần số
    window = np.hamming(len(audio_signal))
    windowed_audio = audio_signal * window
    fft_spectrum = np.abs(np.fft.rfft(windowed_audio))
    frequencies = np.fft.rfftfreq(len(audio_signal), 1/sample_rate)
    
    total_spectrum = np.sum(fft_spectrum)
    
    if total_spectrum == 0:
        return 0, 0, 0
    
    # Spectral Centroid - "trọng tâm" tần số
    spectral_centroid = np.sum(frequencies * fft_spectrum) / total_spectrum
    
    # Spectral Bandwidth - độ rộng phổ
    spectral_bandwidth = np.sqrt(
        np.sum(((frequencies - spectral_centroid) ** 2) * fft_spectrum) / total_spectrum
    )
    
    # Spectral Rolloff - 85% energy cutoff
    cumsum_spectrum = np.cumsum(fft_spectrum)
    rolloff_threshold = 0.85 * total_spectrum
    rolloff_idx = np.where(cumsum_spectrum >= rolloff_threshold)[0]
    spectral_rolloff = frequencies[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
    
    return spectral_centroid, spectral_bandwidth, spectral_rolloff
```

## 🔄 BAR-BASED PROCESSING

### Complete Bar Processing Pipeline
```python
def extract_bar_features(audio_path):
    """Trích xuất features theo từng bar"""
    # 1. Load audio
    sr, audio = wav.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)  # Convert to mono
    audio = audio.astype(np.float32)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)  # Normalize
    
    # 2. Bar parameters
    BPM = 120
    bar_duration = 2.0  # seconds
    bar_samples = int(bar_duration * sr)
    hop_samples = bar_samples // 2  # 50% overlap
    
    # 3. Extract 29 bars
    num_bars = 29
    bar_features = []
    
    for bar_idx in range(num_bars):
        start_sample = bar_idx * hop_samples
        end_sample = start_sample + bar_samples
        bar_audio = audio[start_sample:min(end_sample, len(audio))]
        
        # Padding if bar is too short
        if len(bar_audio) < bar_samples:
            bar_audio = np.pad(bar_audio, (0, bar_samples - len(bar_audio)))
        
        # Calculate 5 features for this bar
        rms = np.sqrt(np.mean(bar_audio ** 2))
        zcr = np.sum(np.abs(np.diff(np.signbit(bar_audio)))) / (2 * len(bar_audio))
        
        # Spectral features
        window = np.hamming(len(bar_audio))
        windowed = bar_audio * window
        spectrum = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(bar_audio), 1/sr)
        
        total_spec = np.sum(spectrum)
        if total_spec > 0:
            centroid = np.sum(freqs * spectrum) / total_spec
            bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / total_spec)
            cumsum_spec = np.cumsum(spectrum)
            rolloff_idx = np.where(cumsum_spec >= 0.85 * total_spec)[0]
            rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
        else:
            centroid = bandwidth = rolloff = 0
        
        bar_feature = [rms, zcr, centroid, bandwidth, rolloff]
        bar_features.append(bar_feature)
    
    # 4. Create final feature vector (151 features)
    feature_vector = []
    
    # Add 29 bars × 5 features = 145 features
    for bar_feat in bar_features:
        feature_vector.extend(bar_feat)
    
    # Add summary vector (average of all bars) = 5 features
    summary = np.mean(bar_features, axis=0)
    feature_vector.extend(summary)
    
    # Add BPM = 1 feature
    feature_vector.append(BPM)
    
    return np.array(feature_vector)
```

## 🔍 SIMILARITY ALGORITHMS

### Cosine Similarity Implementation
```python
def cosine_similarity(vec1, vec2):
    """Tính độ tương đồng cosine"""
    # Normalize vectors
    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
    
    # Calculate cosine similarity
    similarity = np.dot(vec1_norm, vec2_norm)
    return similarity

# Example usage
vec_a = np.array([1, 2, 3, 4, 5])
vec_b = np.array([2, 4, 6, 8, 10])  # Similar direction
similarity = cosine_similarity(vec_a, vec_b)
print(f"Cosine Similarity: {similarity:.4f}")  # High similarity
```

### Euclidean Distance Implementation
```python
def euclidean_distance(vec1, vec2):
    """Tính khoảng cách Euclidean"""
    distance = np.sqrt(np.sum((vec1 - vec2) ** 2))
    return distance

# Example usage
vec_a = np.array([1, 2, 3])
vec_b = np.array([1, 2, 4])  # Close values
distance = euclidean_distance(vec_a, vec_b)
print(f"Euclidean Distance: {distance:.4f}")  # Small distance = similar
```

## 💾 DATABASE OPERATIONS

### Save Features to Database
```python
import sqlite3
import joblib
import io

def save_features_to_db(filename, features, db_path):
    """Lưu features vào database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Serialize features to BLOB
    buffer = io.BytesIO()
    joblib.dump(features, buffer)
    features_blob = buffer.getvalue()
    
    # Extract metadata from filename
    title = filename.split('.')[0]
    genre = filename.split('.')[0] if '.' in filename else 'unknown'
    
    # Insert into database
    cursor.execute("""
        INSERT INTO songs (filename, title, genre, features, feature_count, feature_version)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (filename, title, genre, features_blob, len(features['feature_vector']), "bars_v1"))
    
    conn.commit()
    conn.close()
```

### Load Features from Database
```python
def load_features_from_db(db_path):
    """Load tất cả features từ database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT filename, features FROM songs")
    results = cursor.fetchall()
    
    song_features = []
    for filename, features_blob in results:
        # Deserialize features
        features = joblib.load(io.BytesIO(features_blob))
        song_features.append({
            'filename': filename,
            'feature_vector': features['feature_vector']
        })
    
    conn.close()
    return song_features
```

## 🔄 SEARCH PIPELINE

### Complete Search Function
```python
def search_similar_songs(input_audio_path, db_path, top_k=3):
    """Tìm kiếm bài hát tương tự"""
    # 1. Extract features from input
    input_features = extract_bar_features(input_audio_path)
    
    # 2. Normalize input features
    input_normalized = normalize_features(input_features)
    
    # 3. Load all songs from database
    all_songs = load_features_from_db(db_path)
    
    # 4. Calculate similarities
    similarities = []
    for song in all_songs:
        song_features = song['feature_vector']
        song_normalized = normalize_features(song_features)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(input_normalized, song_normalized)
        similarities.append((song, similarity))
    
    # 5. Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 6. Return top-k results
    return similarities[:top_k]

# Example usage
results = search_similar_songs("input.wav", "database.db", top_k=3)
for i, (song, sim) in enumerate(results):
    print(f"{i+1}. {song['filename']} - {sim:.2%}")
```

## 📊 FEATURE NORMALIZATION

### Min-Max Normalization
```python
def normalize_features(features):
    """Chuẩn hóa features về range [0, 1]"""
    min_val = np.min(features)
    max_val = np.max(features)
    
    # Avoid division by zero
    if max_val - min_val == 0:
        return np.zeros_like(features)
    
    normalized = (features - min_val) / (max_val - min_val)
    return normalized
```

### Z-Score Normalization (Alternative)
```python
def z_score_normalize(features):
    """Chuẩn hóa theo Z-score (mean=0, std=1)"""
    mean = np.mean(features)
    std = np.std(features)
    
    if std == 0:
        return np.zeros_like(features)
    
    normalized = (features - mean) / std
    return normalized
```

## 🧪 TESTING AND VALIDATION

### Quick Test Function
```python
def quick_test_pipeline(test_audio="songs/blues.00000.wav"):
    """Test nhanh pipeline"""
    from audio_processing_pipeline import AudioProcessingPipeline
    
    print("🧪 TESTING PIPELINE...")
    
    # Initialize pipeline
    pipeline = AudioProcessingPipeline()
    print("✅ Pipeline initialized")
    
    # Process test file
    result = pipeline.process(test_audio, top_k=3, verbose=False)
    
    if result['results']:
        print(f"✅ Found {len(result['results'])} similar songs:")
        for i, song in enumerate(result['results']):
            similarity = song.get('similarity', 0)
            print(f"  {i+1}. {song['filename']} - {similarity:.1%}")
    else:
        print("❌ No results found")
    
    print(f"⏱️ Processing time: {result['processing_time']['total']:.2f}s")

# Run test
quick_test_pipeline()
```

## 📈 PERFORMANCE MONITORING

### Timing Decorator
```python
import time
from functools import wraps

def timing_decorator(func):
    """Decorator để đo thời gian execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# Usage
@timing_decorator
def extract_features(audio_path):
    # Your feature extraction code here
    pass
```

## 🎵 DEMO-SPECIFIC CODE

### Streamlit Interface Key Components
```python
import streamlit as st

# File uploader
uploaded_file = st.file_uploader(
    "Chọn file âm thanh (MP3, WAV, OGG)", 
    type=["mp3", "wav", "ogg"]
)

# Processing with spinner
with st.spinner("Đang tìm kiếm các bài hát tương tự..."):
    result = pipeline.process(temp_file_path, top_k=3, verbose=False)

# Display results
if result["results"]:
    st.success(f"✅ Tìm thấy {len(result['results'])} bài hát tương tự!")
    
    for i, song in enumerate(result["results"]):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{i+1}. {song['filename']}**")
        with col2:
            similarity = song.get('similarity', 0)
            st.write(f"{similarity:.1%}")
        
        # Audio player
        song_path = f"songs/{song['filename']}"
        if os.path.exists(song_path):
            st.audio(song_path)
```

---

## 💡 DEBUG TIPS

### Print Feature Statistics
```python
def analyze_features(features):
    """Phân tích thống kê features"""
    print(f"Feature vector length: {len(features)}")
    print(f"Min value: {np.min(features):.6f}")
    print(f"Max value: {np.max(features):.6f}")
    print(f"Mean: {np.mean(features):.6f}")
    print(f"Std: {np.std(features):.6f}")
    
    # Check for NaN or infinite values
    if np.any(np.isnan(features)):
        print("⚠️ Contains NaN values!")
    if np.any(np.isinf(features)):
        print("⚠️ Contains infinite values!")
```

### Database Inspection
```python
def inspect_database(db_path):
    """Kiểm tra thông tin database"""
    import sqlite3
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Count songs
    cursor.execute("SELECT COUNT(*) FROM songs")
    count = cursor.fetchone()[0]
    print(f"Total songs: {count}")
    
    # Check feature versions
    cursor.execute("SELECT feature_version, COUNT(*) FROM songs GROUP BY feature_version")
    versions = cursor.fetchall()
    print("Feature versions:")
    for version, cnt in versions:
        print(f"  {version}: {cnt} songs")
    
    # Check feature counts
    cursor.execute("SELECT feature_count, COUNT(*) FROM songs GROUP BY feature_count")
    counts = cursor.fetchall()
    print("Feature counts:")
    for count, cnt in counts:
        print(f"  {count} features: {cnt} songs")
    
    conn.close()
```

**🎯 Những code examples này sẽ giúp bạn trả lời chi tiết các câu hỏi technical trong buổi vấn đáp!** 