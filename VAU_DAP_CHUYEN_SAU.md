# 🎵 HỆ THỐNG TÌM KIẾM NHẠC TƯƠNG TỰ - TÀI LIỆU VẤN ĐÁP CHUYÊN SÂU

## 📋 TỔNG QUAN HỆ THỐNG

### 🎯 Mục tiêu Chính
Xây dựng hệ thống **Content-Based Music Information Retrieval (CBMIR)** sử dụng kỹ thuật phân tích âm thanh theo từng **bar** với **overlap 50%** để tìm kiếm các bài hát có đặc điểm tương tự.

### 🏗️ Kiến Trúc Tổng Quan
```
Input Audio → Feature Extraction → Database Storage → Similarity Search → Ranked Results
```

### 📊 Thống Kê Hệ Thống
- **Dataset**: 130 bài hát, 10 thể loại (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)
- **Features**: 151 đặc trưng mỗi bài hát
- **Processing Time**: ~1-2 giây cho một query
- **Accuracy**: Tốt cho demo, similar genres cluster together

---

## 🎵 KIẾN THỨC CHUYÊN SÂU VỀ ĐẶC TRƯNG ÂM THANH

### 1. RMS Energy (Root Mean Square Energy)
**Công thức toán học:**
```
RMS = √(1/N × Σ(xi²))
```
**Ý nghĩa vật lý:**
- Đo lường **năng lượng trung bình** của tín hiệu âm thanh
- Tương quan với **loudness** mà tai người nghe được
- **Range**: 0.0 - 1.0 (sau normalization)

**Ứng dụng thực tế:**
- Rock/Metal: RMS cao (0.15-0.25)
- Classical/Ambient: RMS thấp (0.05-0.15)
- Silence: RMS ≈ 0

### 2. Zero-Crossing Rate (ZCR)
**Công thức:**
```
ZCR = 1/2N × Σ|sign(x[n]) - sign(x[n-1])|
```
**Ý nghĩa:**
- Đếm số lần tín hiệu **đổi dấu** trong một đơn vị thời gian
- Phản ánh **texture** của âm thanh (smooth vs rough)

**Phân tích theo loại âm:**
- **Voiced sounds** (vocal): ZCR thấp (~0.02-0.05)
- **Unvoiced sounds** (s, f, h): ZCR cao (~0.15-0.30)  
- **Instrumental**: ZCR trung bình (~0.05-0.15)

### 3. Spectral Centroid
**Định nghĩa:**
"Trọng tâm" của phổ tần số - tần số mà tại đó tập trung nhiều năng lượng nhất.

**Công thức:**
```
SC = Σ(f[k] × X[k]) / Σ(X[k])
```
Trong đó:
- f[k]: Tần số thứ k
- X[k]: Magnitude của FFT tại bin k

**Ý nghĩa âm nhạc:**
- **Brightness**: Cao = sáng, thấp = tối
- **Instrumental timbre**: Piano (cao), Bass guitar (thấp)

### 4. Spectral Bandwidth
**Công thức:**
```
SB = √(Σ((f[k] - SC)² × X[k]) / Σ(X[k]))
```
**Ý nghĩa:**
- Đo **độ rộng phổ tần số** quanh spectral centroid
- Phản ánh **complexity** của harmonics

**Phân tích:**
- **Pure tones**: Bandwidth thấp
- **Noise/Complex sounds**: Bandwidth cao
- **Multi-instrumental**: Bandwidth trung bình đến cao

### 5. Spectral Rolloff
**Định nghĩa:**
Tần số mà tại đó 85% tổng năng lượng phổ được tập trung.

**Algorithm:**
```python
cumulative_energy = cumsum(spectrum)
rolloff_85 = frequencies[cumulative_energy >= 0.85 * total_energy][0]
```

**Ứng dụng:**
- **High-frequency instruments**: Rolloff cao (cymbals, hi-hats)
- **Low-frequency instruments**: Rolloff thấp (bass, cello)

---

## 🔄 PHƯƠNG PHÁP BAR-BASED EXTRACTION

### 🎼 Lý Thuyết Âm Nhạc
**Bar (Measure)**: Đơn vị thời gian cơ bản trong âm nhạc
- **4/4 time signature**: 4 beats per bar
- **Tempo 120 BPM**: 1 beat = 0.5 giây → 1 bar = 2 giây

### 🔧 Technical Implementation

#### Tham Số Cấu Hình
```python
BPM = 120                    # Cố định cho consistency
beats_per_bar = 4           # 4/4 time signature
bar_duration = 60/BPM * 4   # = 2.0 giây
overlap_ratio = 0.5         # 50% overlap
num_bars = 29               # Tối ưu cho 60-second songs
```

#### Thuật Toán Chia Bar
```python
def create_bars(audio_signal, sample_rate):
    bar_samples = int(2.0 * sample_rate)      # 2 giây
    hop_samples = bar_samples // 2            # 1 giây hop
    
    bars = []
    for i in range(29):
        start = i * hop_samples
        end = start + bar_samples
        bar = audio_signal[start:end]
        
        # Padding nếu thiếu data
        if len(bar) < bar_samples:
            bar = np.pad(bar, (0, bar_samples - len(bar)))
        
        bars.append(bar)
    
    return bars
```

### 📊 Ưu Điểm Bar-Based Approach

1. **Temporal Consistency**: Mỗi bar có độ dài cố định (2 giây)
2. **Musical Alignment**: Theo structure tự nhiên của nhạc
3. **Overlap Benefits**: Không bỏ sót thông tin ở ranh giới
4. **Scalability**: Dễ parallel processing từng bar
5. **Detail Capture**: 29 bars × 5 features = 145 temporal features

### 🎯 Feature Vector Final Structure
```
[Bar1_Features] [Bar2_Features] ... [Bar29_Features] [Summary_Features] [BPM]
     5 vals         5 vals            5 vals            5 vals        1 val
     
Total: 29×5 + 5 + 1 = 151 features
```

---

## 💾 CƠ SỞ DỮ LIỆU ĐA PHƯƠNG TIỆN

### 🗄️ Database Schema
```sql
CREATE TABLE songs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,               -- blues.00000.wav
    title TEXT,                          -- blues 00000  
    genre TEXT,                          -- blues
    features BLOB NOT NULL,              -- Serialized feature dict
    feature_count INTEGER,               -- 151
    feature_version TEXT,                -- "bars_v1"
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 🔐 BLOB Storage Strategy

#### Serialization Process
```python
import joblib
import io

# Serialize complex Python object
feature_dict = {
    'bpm': 120,
    'num_bars': 29,
    'bar_features': [...],           # 29 bars × 5 features
    'summary_vector': [...],         # 5 features
    'feature_vector': [...]          # 151 features
}

buffer = io.BytesIO()
joblib.dump(feature_dict, buffer)
features_blob = buffer.getvalue()   # Binary data for BLOB
```

#### Deserialization Process
```python
# Load from BLOB
features = joblib.load(io.BytesIO(features_blob))
feature_vector = features['feature_vector']  # Extract 151-d vector
```

### 📈 Advantages of BLOB Approach

1. **Flexibility**: Store complex nested structures
2. **Efficiency**: Binary format, compressed
3. **Versioning**: Can store different feature versions
4. **Atomicity**: Single transaction per song
5. **Extensibility**: Easy to add new feature types

### 🔍 Indexing Strategy (for scaling)
```sql
-- Metadata indexes for fast filtering
CREATE INDEX idx_genre ON songs(genre);
CREATE INDEX idx_filename ON songs(filename);
CREATE INDEX idx_feature_version ON songs(feature_version);

-- For similarity search, would need:
-- - LSH (Locality Sensitive Hashing) 
-- - Vector databases (Faiss, Pinecone)
-- - Approximate nearest neighbors
```

---

## 🔍 THUẬT TOÁN TÌM KIẾM

### 📐 Cosine Similarity (Primary Method)

#### Mathematical Foundation
```
cos(θ) = (A · B) / (||A|| × ||B||)

Where:
- A, B: Feature vectors (151-dimensional)
- A · B: Dot product
- ||A||: L2 norm of vector A
```

#### Normalization Process
```python
def normalize_vector(v):
    norm = np.sqrt(np.sum(v**2))
    if norm == 0:
        return v
    return v / norm

def cosine_similarity(a, b):
    a_norm = normalize_vector(a)
    b_norm = normalize_vector(b)
    return np.dot(a_norm, b_norm)
```

#### Why Cosine for Audio?
1. **Scale Invariant**: Không bị ảnh hưởng bởi volume differences
2. **Direction Focus**: Quan tâm pattern, không phải magnitude
3. **High Dimensional**: Hiệu quả với 151-D vectors
4. **Robust**: Less sensitive to outlier features

### 📏 Euclidean Distance (Alternative)

#### Formula & Implementation
```python
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

# Converted to similarity (0-1 range)
def euclidean_similarity(a, b):
    dist = euclidean_distance(a, b)
    return 1 / (1 + dist)  # Closer distance → higher similarity
```

#### When to Use Euclidean?
- When **absolute differences** matter
- For **magnitude-sensitive** features
- With **pre-normalized** data

### 🎯 Search Algorithm

#### Complete Pipeline
```python
def search_similar_songs(input_features, database_features, top_k=3):
    similarities = []
    
    # Normalize input
    input_norm = normalize_features(input_features)
    
    # Compare with all songs in database
    for song_id, song_features in database_features:
        song_norm = normalize_features(song_features)
        
        # Calculate similarity
        similarity = cosine_similarity(input_norm, song_norm)
        similarities.append((song_id, similarity))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top-k results
    return similarities[:top_k]
```

#### Time Complexity
- **Feature Extraction**: O(N) where N = audio length
- **Database Search**: O(M×D) where M = #songs, D = feature dims
- **Sorting**: O(M log M)
- **Total**: O(N + M×D + M log M)

For our case: N≈88,200 (2-sec audio), M=130, D=151
→ Very fast, suitable for real-time

---

## 🎛️ FEATURE NORMALIZATION

### 📊 Min-Max Normalization
```python
def min_max_normalize(features):
    """Scale features to [0, 1] range"""
    min_val = np.min(features)
    max_val = np.max(features)
    
    if max_val == min_val:
        return np.zeros_like(features)
    
    return (features - min_val) / (max_val - min_val)
```

**Advantages:**
- Preserves relationships
- All features same scale
- Interpretable [0,1] range

### 📈 Z-Score Normalization (Alternative)
```python
def z_score_normalize(features):
    """Standardize to mean=0, std=1"""
    mean = np.mean(features)
    std = np.std(features)
    
    if std == 0:
        return np.zeros_like(features)
    
    return (features - mean) / std
```

**When to use:**
- Features follow normal distribution
- Want to remove scale effects
- Preserve relative distances

### 🎯 Why Normalization is Critical

**Problem without normalization:**
```
RMS Energy: 0.1 - 0.3 range
Spectral Centroid: 1000 - 5000 Hz range
→ Centroid dominates similarity calculation
```

**Solution with normalization:**
```
All features: 0.0 - 1.0 range
→ Equal weight in similarity calculation
```

---

## 🚀 PERFORMANCE OPTIMIZATION

### ⚡ Real-time Processing Techniques

1. **Vectorized Operations**
```python
# Slow: Loop-based
for i in range(len(audio)):
    rms += audio[i] ** 2

# Fast: Vectorized
rms = np.sqrt(np.mean(audio ** 2))
```

2. **Pre-computed FFT Windows**
```python
# Pre-compute Hamming window
window = np.hamming(bar_length)  # Compute once

# Use in each bar
for bar in bars:
    windowed = bar * window      # Fast element-wise multiply
    spectrum = np.fft.rfft(windowed)
```

3. **Database Connection Pooling**
```python
# Connection reuse instead of open/close per query
class DatabaseManager:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
    
    def search(self, features):
        # Reuse existing connection
        cursor = self.conn.cursor()
        # ... search logic
```

### 📊 Profiling Results
```
Feature Extraction: ~0.8s (80% of total time)
├── Audio loading: ~0.1s
├── Bar creation: ~0.1s  
├── FFT computation: ~0.4s
└── Feature calculation: ~0.2s

Database Search: ~0.2s (20% of total time)
├── Loading features: ~0.1s
└── Similarity calculation: ~0.1s
```

---

## 🎯 VALIDATION & TESTING

### 🧪 Accuracy Evaluation

#### Test Cases
```python
test_cases = {
    'blues.00000.wav': ['blues.00001.wav', 'blues.00013.wav'],
    'classical.00000.wav': ['classical.00001.wav', 'classical.00016.wav'],
    'rock.00000.wav': ['rock.00001.wav', 'rock.00014.wav']
}

def evaluate_accuracy():
    correct_predictions = 0
    total_tests = len(test_cases)
    
    for input_song, expected_similar in test_cases.items():
        results = search_similar_songs(input_song, top_k=3)
        
        # Check if at least 1 expected song in top-3 results
        found_similar = any(song in [r['filename'] for r in results] 
                          for song in expected_similar)
        
        if found_similar:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_tests
    return accuracy
```

#### Cross-validation Strategy
1. **Leave-one-out**: Remove 1 song, test against remaining 129
2. **Genre-based split**: Train on 8 genres, test on 2
3. **Temporal split**: Use different recording sessions

### 📈 Performance Metrics

#### Precision & Recall
```python
def calculate_precision_recall(predicted, ground_truth):
    true_positives = len(set(predicted) & set(ground_truth))
    
    precision = true_positives / len(predicted) if predicted else 0
    recall = true_positives / len(ground_truth) if ground_truth else 0
    
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score
```

#### Mean Reciprocal Rank (MRR)
```python
def calculate_mrr(results_list):
    """Measure how high relevant results appear in ranking"""
    reciprocal_ranks = []
    
    for results in results_list:
        for rank, item in enumerate(results, 1):
            if item['is_relevant']:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks)
```

---

## 🔮 FUTURE IMPROVEMENTS

### 🎵 Advanced Audio Features

#### 1. MFCC (Mel-Frequency Cepstral Coefficients)
```python
def extract_mfcc(audio, sr, n_mfcc=13):
    """Extract perceptually-motivated features"""
    # Mel-scale frequency banks
    mel_filters = create_mel_filterbank(sr)
    
    # Apply filters to spectrum
    mel_spectrum = np.dot(mel_filters, np.abs(np.fft.rfft(audio))**2)
    
    # Log and DCT transform
    log_mel = np.log(mel_spectrum + 1e-8)
    mfcc = dct(log_mel)[:n_mfcc]
    
    return mfcc
```

#### 2. Chroma Features
```python
def extract_chroma(audio, sr):
    """Extract pitch class distribution"""
    # Map frequencies to 12 pitch classes (C, C#, D, ...)
    chroma = np.zeros(12)
    
    freqs = np.fft.rfftfreq(len(audio), 1/sr)
    spectrum = np.abs(np.fft.rfft(audio))
    
    for i, freq in enumerate(freqs):
        if freq > 0:
            # Convert frequency to pitch class
            pitch_class = int(np.round(12 * np.log2(freq / 440.0))) % 12
            chroma[pitch_class] += spectrum[i]
    
    return chroma / np.sum(chroma)  # Normalize
```

#### 3. Tonnetz (Harmonic Network)
```python
def extract_tonnetz(chroma):
    """Extract tonal centroid features"""
    # Define harmonic change vectors
    r1 = np.array([1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0])  # Perfect fifths
    r2 = np.array([0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1])  # Minor thirds
    r3 = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]) # Major thirds
    
    tonnetz = np.array([
        np.dot(chroma, r1),
        np.dot(chroma, r2), 
        np.dot(chroma, r3)
    ])
    
    return tonnetz
```

### 🤖 Machine Learning Enhancements

#### 1. Deep Audio Embeddings
```python
# Using pre-trained models like VGGish, OpenL3, or Wav2Vec2
import tensorflow as tf

def extract_deep_features(audio_path):
    """Extract high-level semantic features"""
    model = tf.keras.models.load_model('vggish_model.h5')
    
    # Preprocess audio to model format
    audio_preprocessed = preprocess_for_vggish(audio_path)
    
    # Extract 128-dimensional embeddings
    embeddings = model.predict(audio_preprocessed)
    
    return embeddings.flatten()
```

#### 2. Metric Learning
```python
# Learn optimal distance metric for music similarity
from sklearn.neighbors import NeighborhoodComponentsAnalysis

def learn_similarity_metric(features, labels):
    """Learn transformation that improves similarity"""
    nca = NeighborhoodComponentsAnalysis(n_components=50)
    
    # Transform features to optimize k-NN classification
    features_transformed = nca.fit_transform(features, labels)
    
    return nca, features_transformed
```

### 🏗️ Scalability Improvements

#### 1. Approximate Nearest Neighbors
```python
import faiss

def build_faiss_index(features):
    """Build fast similarity search index"""
    dimension = features.shape[1]
    
    # Create FAISS index
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine)
    
    # Add vectors to index
    index.add(features.astype('float32'))
    
    return index

def search_with_faiss(index, query_vector, k=10):
    """Fast approximate search"""
    similarities, indices = index.search(
        query_vector.reshape(1, -1).astype('float32'), 
        k
    )
    return similarities[0], indices[0]
```

#### 2. Hierarchical Clustering
```python
from sklearn.cluster import AgglomerativeClustering

def create_music_hierarchy(features, n_clusters=20):
    """Create hierarchical organization of songs"""
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='ward'
    )
    
    cluster_labels = clustering.fit_predict(features)
    
    # Create cluster-based search
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)
    
    return clusters
```

---

## 📚 CÂU HỎI VẤN ĐÁP CHUYÊN SÂU

### 🔬 Technical Deep Dive Questions

**Q: Giải thích chi tiết process trích xuất spectral centroid?**

A: Spectral centroid được tính qua các bước:

1. **Windowing**: Áp dụng Hamming window để giảm spectral leakage
```python
window = np.hamming(len(audio_segment))
windowed_signal = audio_segment * window
```

2. **FFT Transform**: Chuyển từ time domain sang frequency domain
```python
spectrum = np.abs(np.fft.rfft(windowed_signal))
frequencies = np.fft.rfftfreq(len(windowed_signal), 1/sample_rate)
```

3. **Weighted Average**: Tính trọng tâm tần số
```python
total_energy = np.sum(spectrum)
if total_energy > 0:
    centroid = np.sum(frequencies * spectrum) / total_energy
```

4. **Physical Meaning**: Centroid cao → âm thanh "sáng", nhiều high-frequency components

**Q: Tại sao overlap 50% cho bars? Có thể optimize không?**

A: 50% overlap balance giữa detail và efficiency:

**Advantages của 50%:**
- Ensure không miss information ở bar boundaries  
- Smooth temporal transitions
- Standard trong audio processing literature

**Alternative overlaps:**
- **25% overlap**: Faster, ít redundancy, có thể miss transients
- **75% overlap**: More detail, nhưng 3x slower processing
- **Variable overlap**: Adaptive dựa trên onset detection

**Optimization strategy:**
```python
def adaptive_overlap(audio, onset_times):
    """Vary overlap based on musical structure"""
    overlaps = []
    for i in range(len(onset_times)-1):
        segment_length = onset_times[i+1] - onset_times[i]
        
        if segment_length < 1.0:  # Short segments
            overlap = 0.75  # More overlap
        elif segment_length > 3.0:  # Long segments  
            overlap = 0.25  # Less overlap
        else:
            overlap = 0.50  # Standard
            
        overlaps.append(overlap)
    
    return overlaps
```

**Q: Database có handle concurrent access không? Scalability?**

A: Current SQLite setup basic, có limitations:

**Current Limitations:**
- Single-writer, multiple-readers
- File-based locking
- No distributed capability

**Production Improvements:**
```python
# 1. Connection pooling
class DatabasePool:
    def __init__(self, db_path, pool_size=10):
        self.pool = queue.Queue()
        for _ in range(pool_size):
            conn = sqlite3.connect(db_path, check_same_thread=False)
            self.pool.put(conn)
    
    def get_connection(self):
        return self.pool.get()
    
    def return_connection(self, conn):
        self.pool.put(conn)

# 2. Read replicas
class ReadWriteDatabase:
    def __init__(self, write_db, read_dbs):
        self.write_db = write_db
        self.read_dbs = read_dbs  # List of read replicas
        self.read_index = 0
    
    def read_query(self, query):
        # Load balance across read replicas
        db = self.read_dbs[self.read_index % len(self.read_dbs)]
        self.read_index += 1
        return db.execute(query)

# 3. Distributed approach (PostgreSQL + Redis)
import redis
import psycopg2

class DistributedMusicDB:
    def __init__(self):
        self.postgres = psycopg2.connect(...)  # Metadata
        self.redis = redis.Redis(...)          # Feature vectors cache
    
    def store_song(self, song_data, features):
        # Store metadata in PostgreSQL
        self.postgres.execute(
            "INSERT INTO songs (filename, genre) VALUES (%s, %s)",
            (song_data['filename'], song_data['genre'])
        )
        
        # Cache features in Redis
        self.redis.set(
            f"features:{song_data['id']}", 
            pickle.dumps(features)
        )
```

**Q: Làm sao evaluate quality của similarity results?**

A: Multiple evaluation strategies:

**1. Objective Metrics:**
```python
def evaluate_retrieval_quality(test_queries):
    metrics = {
        'precision_at_k': [],
        'recall_at_k': [],
        'map_score': [],  # Mean Average Precision
        'mrr_score': []   # Mean Reciprocal Rank
    }
    
    for query in test_queries:
        results = search_similar_songs(query['audio'])
        ground_truth = query['similar_songs']
        
        # Calculate metrics
        p_at_k = precision_at_k(results, ground_truth, k=5)
        r_at_k = recall_at_k(results, ground_truth, k=5)
        
        metrics['precision_at_k'].append(p_at_k)
        metrics['recall_at_k'].append(r_at_k)
    
    return {k: np.mean(v) for k, v in metrics.items()}
```

**2. Human Evaluation:**
```python
def human_evaluation_interface():
    """Web interface for human similarity rating"""
    # Present query song + top-5 results
    # Ask humans to rate similarity 1-5 scale
    # Calculate inter-annotator agreement
    # Use as ground truth for model evaluation
```

**3. Cross-validation:**
```python
def genre_based_validation():
    """Test generalization across genres"""
    genres = ['blues', 'classical', 'rock', 'jazz']
    
    for test_genre in genres:
        train_songs = [s for s in all_songs if s.genre != test_genre]
        test_songs = [s for s in all_songs if s.genre == test_genre]
        
        # Build index from train songs
        train_features = extract_features(train_songs)
        
        # Test on test songs
        for test_song in test_songs:
            results = search_in_features(test_song, train_features)
            # Evaluate if results make sense
```

**Q: Feature vector 151-D có curse of dimensionality không?**

A: Có considerations, nhưng manageable:

**Curse of Dimensionality Issues:**
- Distance metrics become less discriminative
- Nearest neighbors tend to be equally far
- Need exponentially more data

**Why 151-D works cho music:**
1. **Structured dimensions**: Không random, có musical meaning
2. **Redundancy**: Các bars có correlation, giảm effective dimensionality  
3. **Domain-specific**: Audio features có natural clustering

**Mitigation strategies:**
```python
# 1. Dimensionality reduction
from sklearn.decomposition import PCA

def reduce_dimensions(features, n_components=50):
    """Reduce to most important dimensions"""
    pca = PCA(n_components=n_components)
    features_reduced = pca.fit_transform(features)
    
    # Check explained variance
    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"Retained {explained_var:.2%} of variance")
    
    return features_reduced, pca

# 2. Feature selection
from sklearn.feature_selection import SelectKBest, mutual_info_regression

def select_best_features(features, labels, k=50):
    """Select most informative features"""
    selector = SelectKBest(mutual_info_regression, k=k)
    features_selected = selector.fit_transform(features, labels)
    
    selected_indices = selector.get_support(indices=True)
    return features_selected, selected_indices

# 3. Locality Sensitive Hashing
import numpy as np

class LSH:
    """Approximate nearest neighbors for high-D data"""
    def __init__(self, n_hashes=10, n_bits=8):
        self.n_hashes = n_hashes
        self.n_bits = n_bits
        self.hash_functions = []
        
    def fit(self, data):
        """Create random projection hash functions"""
        d = data.shape[1]
        
        for _ in range(self.n_hashes):
            # Random projection matrix
            projection = np.random.randn(d, self.n_bits)
            self.hash_functions.append(projection)
    
    def hash_vector(self, vector):
        """Convert vector to hash codes"""
        hashes = []
        for projection in self.hash_functions:
            # Project and threshold
            projected = np.dot(vector, projection)
            hash_code = (projected > 0).astype(int)
            hashes.append(hash_code)
        return hashes
```

---

## 🎯 DEMO PRESENTATION STRATEGY

### 🎪 Live Demo Flow

1. **Opening Hook** (30 seconds)
   - "Có bao giờ bạn nghe một bài nhạc và muốn tìm những bài tương tự?"
   - Show demo interface trước

2. **Technical Overview** (2 minutes)
   - Quick architecture diagram
   - Highlight key innovations: bar-based, 151 features

3. **Live Demo** (3 minutes)
   - Upload blues song → show blues results
   - Upload classical → show classical results  
   - Upload rock → show mixed results (interesting case)

4. **Deep Dive** (5 minutes)
   - Code walkthrough của key functions
   - Show feature extraction process
   - Explain similarity calculation

5. **Q&A Preparation** (remaining time)
   - Ready for technical questions
   - Have backup examples prepared

### 🎯 Key Messages to Emphasize

1. **Innovation**: Bar-based approach với overlap
2. **Robustness**: Multiple complementary features
3. **Performance**: Real-time processing
4. **Scalability**: Clear path to production scale
5. **Accuracy**: Good results cho demo dataset

### 🔧 Backup Plans

**If demo fails:**
- Have pre-recorded screen capture
- Static screenshots của results
- Code examples có thể run offline

**If technical questions too deep:**
- Acknowledge limitations honestly
- Show improvement roadmap
- Redirect to working components

---

**🎵 Good luck với presentation! Remember: Be confident, explain clearly, và show passion for the technical challenges! 🎵** 