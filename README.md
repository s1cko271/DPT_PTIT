# 🎵 HỆ THỐNG TÌM KIẾM NHẠC TƯƠNG TỰ

Hệ thống **Content-Based Music Information Retrieval (CBMIR)** sử dụng kỹ thuật phân tích âm thanh theo từng **bar** với **overlap 50%** để tìm kiếm các bài hát có đặc điểm tương tự.

## 🎯 **CÁCH HOẠT ĐỘNG TỔNG QUAN**

```
Âm thanh đầu vào → Phân đoạn Bar → Trích xuất đặc trưng → Lưu trữ CSDL → Tìm kiếm tương tự → Kết quả xếp hạng
```

1. **Tải lên nhạc** → Giao diện web nhận file MP3/WAV/OGG
2. **Phân tích bars** → Chia thành 29 đoạn nhạc (2 giây/đoạn, chồng lấp 50%)  
3. **Trích xuất đặc trưng** → 5 đặc trưng/bar + tổng hợp = 151 đặc trưng tổng cộng
4. **So sánh cơ sở dữ liệu** → Độ tương đồng cosine với 130 bài hát có sẵn
5. **Trả kết quả** → Top-3 bài hát tương tự nhất với trình phát âm thanh

## 📂 TỔNG QUAN CÁC FILE PYTHON

Hệ thống gồm **4 file Python chính** làm việc cùng nhau để tạo thành một pipeline hoàn chỉnh:

### 🎯 **1. `streamlit_demo.py` - FILE CHÍNH (GIAO DIỆN WEB)**

**Chức năng chính:**
- **Giao diện web** để người dùng tải lên file nhạc và xem kết quả
- **Demo trực quan** của hệ thống tìm kiếm nhạc tương tự
- **Điểm kết nối** giữa người dùng và toàn bộ hệ thống

**Cách hoạt động:**
1. Người dùng truy cập giao diện web
2. Tải lên file nhạc (MP3, WAV, OGG) 
3. Hệ thống tự động:
   - Trích xuất đặc trưng từ file đã tải lên
   - Tìm kiếm 3 bài hát tương tự nhất
   - Hiển thị kết quả với trình phát âm thanh

**Để chạy:**
```bash
streamlit run streamlit_demo.py
```

### 🔍 **2. `audio_search.py` - THUẬT TOÁN TÌM KIẾM**

**Chức năng:**
- **Tính toán độ tương đồng** giữa các bài hát sử dụng **Độ tương đồng Cosine**
- **Chuẩn hóa dữ liệu** để so sánh chính xác
- **Tìm kiếm** và **xếp hạng** các bài hát tương tự

**Các hàm chính:**
- `cosine_similarity()` - Tính độ tương đồng cosine
- `normalize_features()` - Chuẩn hóa vector đặc trưng
- `search_similar_songs()` - Tìm kiếm bài hát tương tự

**Công thức Độ tương đồng Cosine:**
```
cos(θ) = (A · B) / (||A|| × ||B||)
```

### 🎵 **3. `modified_feature_extraction.py` - TRÍCH XUẤT ĐẶC TRƯNG**

**Chức năng:**
- **Trích xuất 151 đặc trưng** từ mỗi bài hát
- **Phân tích theo bars** (29 đoạn nhạc, mỗi đoạn 2 giây với overlap 50%)
- **Lưu trữ vào database** SQLite

**Quy trình trích xuất:**
1. Chia file nhạc thành 29 bars (đoạn nhạc)
2. Mỗi bar trích xuất 5 đặc trưng:
   - **RMS Energy** (năng lượng)
   - **Zero-Crossing Rate** (tần số đổi dấu)
   - **Spectral Centroid** (trọng tâm phổ tần)
   - **Spectral Bandwidth** (độ rộng phổ)
   - **Spectral Rolloff** (điểm giới hạn phổ)
3. Tạo vector tổng hợp: **29×5 + 5 + 1 = 151 đặc trưng**

**Cấu trúc Vector Đặc trưng:**
```
[Đặc_trưng_Bar1] [Đặc_trưng_Bar2] ... [Đặc_trưng_Bar29] [Đặc_trưng_Tổng_hợp] [BPM]
     5 giá trị         5 giá trị            5 giá trị            5 giá trị        1 giá trị
```

### ⚡ **4. `audio_processing_pipeline.py` - PIPELINE TỔNG HỢP**

**Chức năng:**
- **Kết nối tất cả mô-đun** thành một quy trình hoàn chỉnh
- **Class `AudioProcessingPipeline`** điều phối toàn bộ quá trình
- **Theo dõi thời gian xử lý** từng bước

**Quy trình 4 bước:**
1. **Nhận âm thanh** - Kiểm tra file đầu vào
2. **Trích xuất đặc trưng** - Phân tích âm thanh (29 bars × 5 đặc trưng)
3. **So sánh đặc trưng** - Tìm kiếm trong cơ sở dữ liệu sử dụng độ tương đồng Cosine
4. **Trả kết quả** - Định dạng và trả về top-k bài hát tương tự



## 🔄 MỐI QUAN HỆ GIỮA CÁC FILE

```
streamlit_demo.py (GIAO DIỆN)
       ↓
audio_processing_pipeline.py (ĐIỀU PHỐI)
       ↓                    ↓
modified_feature_extraction.py  audio_search.py
     (TRÍCH XUẤT)              (TÌM KIẾM)
       ↓                    ↓
   DATABASE.db ←---------DATABASE.db
```

## 🎯 ĐẶC ĐIỂM KỸ THUẬT

### 📊 **Bộ Dữ Liệu**
- **130 bài hát**, **10 thể loại**: blues, cổ điển, đồng quê, disco, hiphop, jazz, metal, pop, reggae, rock
- **151 đặc trưng** mỗi bài hát
- **Thời gian xử lý**: ~1-2 giây cho một truy vấn

### 🎼 **Phân Tích Theo Bar**
- **BPM**: 120 (cố định để đảm bảo tính nhất quán)
- **Ký hiệu nhịp**: 4/4  
- **Thời lượng Bar**: 2 giây
- **Chồng lấp**: 50%
- **Số lượng Bars**: 29

### 🔍 **Phương Pháp Tính Tương Đồng**
- **Độ tương đồng Cosine** - đo độ tương đồng về hướng
- **Bất biến theo tỷ lệ** - không bị ảnh hưởng bởi âm lượng
- **Tập trung vào mẫu** thay vì độ lớn

## 🚀 CÁCH CHẠY HỆ THỐNG

### **Chạy Demo (Cách chính):**
```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Chạy web interface
streamlit run streamlit_demo.py
```

### **Chạy từng mô-đun riêng lẻ:**
```bash
# Kiểm tra trích xuất đặc trưng
python modified_feature_extraction.py

# Kiểm tra tìm kiếm
python audio_search.py

# Kiểm tra pipeline
python audio_processing_pipeline.py
```

## 💾 CƠ SỞ DỮ LIỆU - CHI TIẾT KỸ THUẬT

### 🗄️ **Kiến Trúc Cơ Sở Dữ Liệu**

Hệ thống sử dụng **SQLite** làm cơ sở dữ liệu chính với thiết kế tối ưu cho việc lưu trữ và truy xuất đặc trưng âm thanh.

**File cơ sở dữ liệu:** `./database/music_features_new.db`

### 📋 **Lược Đồ Cơ Sở Dữ Liệu**
```sql
CREATE TABLE songs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,    -- ID tự động tăng
    filename TEXT NOT NULL,                  -- Tên file (ví dụ: blues.00000.wav)
    title TEXT,                             -- Tên bài hát (ví dụ: blues 00000)
    genre TEXT,                             -- Thể loại (blues, rock, jazz...)
    features BLOB NOT NULL,                 -- Dữ liệu đặc trưng (binary)
    feature_count INTEGER,                  -- Số lượng đặc trưng (151)
    feature_version TEXT,                   -- Phiên bản đặc trưng ("bars_v1")
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- Thời gian tạo
);
```

### 🔐 **Lưu Trữ Đặc Trưng BLOB - Cách Lưu Trữ Phức Tạp**

#### **Cấu trúc dữ liệu trong BLOB:**
```python
feature_dict = {
    'bpm': 120,                           # Tempo cố định
    'num_bars': 29,                       # Số lượng bars
    'bar_duration': 2.0,                  # Độ dài mỗi bar (giây)
    'bar_features': [                     # List 29 arrays
        [rms, zcr, sc, sb, sr],          # Bar 1: 5 đặc trưng
        [rms, zcr, sc, sb, sr],          # Bar 2: 5 đặc trưng
        ...                               # ... 27 bars khác
        [rms, zcr, sc, sb, sr]           # Bar 29: 5 đặc trưng
    ],
    'summary_vector': [rms, zcr, sc, sb, sr],  # Vector tổng hợp
    'feature_vector': [151 giá_trị]        # Vector cuối cùng để so sánh
}
```

#### **Quá trình Serialization:**
```python
import joblib
import io

# Chuyển đổi từ đối tượng Python → Dữ liệu nhị phân
buffer = io.BytesIO()
joblib.dump(feature_dict, buffer)
features_blob = buffer.getvalue()  # Dữ liệu nhị phân để lưu vào BLOB

# Lưu vào cơ sở dữ liệu
cursor.execute("""
    INSERT INTO songs (filename, title, genre, features, feature_count, feature_version) 
    VALUES (?, ?, ?, ?, ?, ?)
""", (filename, title, genre, features_blob, 151, "bars_v1"))
```

#### **Quá trình Deserialization:**
```python
# Lấy từ cơ sở dữ liệu → Đối tượng Python
cursor.execute("SELECT features FROM songs WHERE id = ?", (song_id,))
features_blob = cursor.fetchone()[0]

# Chuyển đổi dữ liệu nhị phân → Đối tượng Python
features = joblib.load(io.BytesIO(features_blob))
feature_vector = features['feature_vector']  # Lấy vector 151-D
```

### 📊 **Thống Kê Cơ Sở Dữ Liệu Hiện Tại**

```python
# Ví dụ dữ liệu thực tế:
130 bài hát tổng cộng:
├── blues: 12 bài hát      (blues.00000.wav → blues.00099.wav)
├── cổ điển: 11 bài hát  (classical.00000.wav → classical.00099.wav)  
├── đồng quê: 15 bài hát    (country.00000.wav → country.00099.wav)
├── disco: 14 bài hát      (disco.00000.wav → disco.00099.wav)
├── hiphop: 15 bài hát     (hiphop.00000.wav → hiphop.00099.wav)
├── jazz: 14 bài hát       (jazz.00000.wav → jazz.00099.wav)
├── metal: 12 bài hát      (metal.00000.wav → metal.00099.wav)
├── pop: 13 bài hát        (pop.00000.wav → pop.00099.wav)
├── reggae: 12 bài hát     (reggae.00000.wav → reggae.00099.wav)
└── rock: 12 bài hát       (rock.00000.wav → rock.00099.wav)

Tổng dung lượng: ~2.5MB
Trung bình mỗi bài hát: ~20KB (bao gồm siêu dữ liệu + 151 đặc trưng)
```

### ⚡ **Hiệu Suất & Lập Chỉ Mục**

#### **Hiệu Suất Truy Vấn:**
```sql
-- Truy vấn nhanh theo thể loại
CREATE INDEX idx_genre ON songs(genre);

-- Truy vấn nhanh theo tên file  
CREATE INDEX idx_filename ON songs(filename);

-- Truy vấn nhanh theo phiên bản đặc trưng
CREATE INDEX idx_feature_version ON songs(feature_version);
```

#### **Quy Trình Tìm Kiếm Tương Đồng:**
```python
def quy_trinh_tim_kiem():
    # 1. Tải tất cả đặc trưng (130 bản ghi) - ~0.05s
    tat_ca_dac_trung = load_all_features_from_db()
    
    # 2. So sánh với vector đầu vào (151-D × 130 bài hát) - ~0.1s  
    do_tuong_dong = calculate_cosine_similarities(input_vector, tat_ca_dac_trung)
    
    # 3. Sắp xếp và lấy top-k - ~0.01s
    ket_qua_hang_dau = sorted(do_tuong_dong)[:top_k]
    
    # Tổng cộng: ~0.16s cho các thao tác cơ sở dữ liệu
```

### 🔧 **Các Hàm Quản Lý Cơ Sở Dữ Liệu**

#### **Thêm bài hát mới:**
```python
def add_song_to_database(audio_path, db_path):
    features = extract_features(audio_path)           # ~0.8s
    genre = detect_genre_from_filename(audio_path)    # ~0.001s
    save_to_database(features, genre, db_path)        # ~0.05s
```

#### **Truy xuất đặc trưng:**
```python
def get_song_features(song_id, db_path):
    features = retrieve_features(db_path, song_id=song_id)
    return features['feature_vector']  # Vector 151-D
```

#### **Thống kê cơ sở dữ liệu:**
```python
def thong_ke_co_so_du_lieu(db_path):
    tong_bai_hat = count_total_songs()
    the_loai = get_genre_distribution() 
    do_tuong_dong_tb = calculate_avg_similarity()
    return thong_ke
```

### 💡 **Ưu Điểm của Thiết Kế Cơ Sở Dữ Liệu**

1. **Lưu trữ linh hoạt**: BLOB cho phép lưu các đối tượng Python phức tạp
2. **Gọn nhẹ**: Nén bằng joblib, tiết kiệm không gian
3. **Truy xuất nhanh**: Chỉ mục trên siêu dữ liệu, không cần giải mã để lọc
4. **Kiểm soát phiên bản**: Theo dõi phiên bản đặc trưng qua `feature_version`
5. **Có thể mở rộng**: Dễ thêm bài hát mới và loại đặc trưng mới

### 🚀 **Cân Nhắc Mở Rộng**

Với cơ sở dữ liệu lớn hơn (>10,000 bài hát), nên cân nhắc:
- **Cơ sở dữ liệu vector** (Faiss, Pinecone) cho tìm kiếm tương đồng
- **Thuật toán láng giềng gần đúng**  
- **Phân vùng cơ sở dữ liệu** theo thể loại
- **Cơ chế bộ nhớ đệm** cho các truy vấn thường xuyên

## 🎯 TẠI SAO CẦN NHIỀU FILE?

1. **Tách biệt mối quan tâm** - Mỗi file có trách nhiệm riêng
2. **Tính mô-đun** - Dễ bảo trì và kiểm tra từng phần  
3. **Tính tái sử dụng** - Các mô-đun có thể dùng lại
4. **Khả năng mở rộng** - Dễ mở rộng và tối ưu từng phần

## 📈 HIỆU SUẤT

**Thời gian xử lý:**
- **Trích xuất đặc trưng**: ~0.8s (80% tổng thời gian)
- **Tìm kiếm cơ sở dữ liệu**: ~0.2s (20% tổng thời gian)
- **Tổng cộng**: ~1.0s cho một truy vấn

**File quan trọng nhất:** `streamlit_demo.py` - đây là điểm khởi đầu và giao diện chính mà người dùng tương tác! 