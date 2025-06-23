# Hệ Thống Tìm Kiếm Nhạc Dựa Trên Đặc Trưng Âm Thanh

Hệ thống này cho phép trích xuất đặc trưng âm thanh từ các file nhạc và tìm kiếm các bài hát tương tự dựa trên đặc trưng này.

## Đặc trưng âm thanh được sử dụng

Hệ thống trích xuất 7 đặc trưng âm thanh chính:

1. **RMS Energy**: Đo độ lớn của tín hiệu âm thanh 
2. **Zero-Crossing Rate**: Đo tần suất tín hiệu chuyển đổi từ dương sang âm và ngược lại
3. **Spectral Centroid**: Đo trọng tâm của phổ - liên quan đến "độ sáng" của âm thanh
4. **Spectral Bandwidth**: Đo độ rộng của phổ - liên quan đến "độ dày" của âm thanh
5. **Spectral Rolloff**: Đo tần số mà phía dưới đó tập trung 85% năng lượng của phổ
6. **Average Energy**: Đo năng lượng trung bình của tín hiệu âm thanh
7. **BPM (Tempo)**: Ước lượng nhịp độ của bài hát bằng phương pháp tự tương quan

## Cài đặt

1. Clone repository về máy:
```
git clone <repository-url>
cd <repository-directory>
```

2. Cài đặt các thư viện cần thiết:
```
pip install numpy scipy scikit-learn joblib mutagen tqdm matplotlib
```

## Sử dụng

### Xây dựng cơ sở dữ liệu
Trước khi sử dụng, bạn cần xây dựng cơ sở dữ liệu từ thư mục nhạc của bạn:

```
python music_search.py build path/to/your/music/folder
```

Lệnh này sẽ:
1. Quét qua tất cả các file nhạc (định dạng .mp3, .wav, .ogg) trong thư mục
2. Trích xuất đặc trưng âm thanh từ mỗi file
3. Lưu đặc trưng này vào cơ sở dữ liệu SQLite (mặc định là `music.db`)

### Tìm kiếm nhạc tương tự

Sau khi đã xây dựng cơ sở dữ liệu, bạn có thể tìm kiếm nhạc tương tự:

```
python music_search.py search path/to/query/song.mp3
```

Mặc định, chương trình sẽ trả về 3 bài hát tương tự nhất. Bạn có thể thay đổi số lượng kết quả:

```
python music_search.py search path/to/query/song.mp3 --num 5
```

### Các tùy chọn khác

- Chỉ định file cơ sở dữ liệu:
  ```
  python music_search.py search path/to/query/song.mp3 --db custom_database.db
  ```

## Hoạt động

1. Hệ thống tự động phân tích và ước lượng BPM (tempo) của bài hát 
2. Dựa trên BPM, hệ thống chia mỗi bài hát thành các frame tương ứng với độ dài một bar nhạc (4 nhịp)
3. Trích xuất đặc trưng âm thanh từ mỗi frame
4. Tính trung bình các đặc trưng trên tất cả các frame để tạo vector đặc trưng cho bài hát
5. Khi tìm kiếm, tính độ tương tự (cosine similarity) giữa vector đặc trưng của bài hát truy vấn và tất cả các bài hát trong cơ sở dữ liệu
6. Trả về các bài hát có độ tương tự cao nhất

## Các cải tiến

Hệ thống sử dụng các kỹ thuật xử lý tín hiệu hiện đại:

1. **Ước lượng Tempo tự động**: Sử dụng kỹ thuật tự tương quan (autocorrelation) trên đường bao năng lượng để phát hiện nhịp độ
2. **Phân tích theo Bar**: Chia nhạc theo đơn vị bar (4 nhịp) thay vì chia theo độ dài cố định, giúp bắt được cấu trúc nhạc tốt hơn
3. **Tính toán FFT hiệu quả**: Sử dụng Fast Fourier Transform để phân tích phổ tần số
4. **Lọc tín hiệu tiên tiến**: Áp dụng bộ lọc Butterworth để xử lý tín hiệu trước khi phân tích

# Music Feature Extraction Formulas

This document outlines the key formulas and calculations used in the music feature extraction system.

## 1. Spectral Features

### 1.1 Spectral Centroid
The spectral centroid represents the "brightness" of a sound and is calculated as the weighted mean of the frequencies present in the signal.

```python
spectral_centroid = np.sum(frequencies * magnitudes) / np.sum(magnitudes)
```

### 1.2 Spectral Bandwidth
Spectral bandwidth measures the spread of the spectrum around its centroid.

```python
spectral_bandwidth = np.sqrt(np.sum(((frequencies - spectral_centroid) ** 2) * magnitudes) / np.sum(magnitudes))
```

### 1.3 Spectral Rolloff
Spectral rolloff is the frequency below which a specified percentage (default 85%) of the total spectral energy lies.

```python
rolloff_threshold = 0.85 * np.sum(magnitudes)
cumulative_sum = np.cumsum(magnitudes)
spectral_rolloff = frequencies[np.where(cumulative_sum >= rolloff_threshold)[0][0]]
```

### 1.4 Spectral Flatness
Spectral flatness measures how noise-like vs. tone-like a sound is.

```python
spectral_flatness = np.exp(np.mean(np.log(magnitudes + 1e-10))) / np.mean(magnitudes)
```

## 2. Temporal Features

### 2.1 Zero Crossing Rate
Measures how many times the signal changes from positive to negative or vice versa.

```python
zero_crossings = np.sum(np.diff(np.signbit(signal)))
zero_crossing_rate = zero_crossings / (len(signal) - 1)
```

### 2.2 Root Mean Square Energy
Measures the average power of the signal.

```python
rms_energy = np.sqrt(np.mean(signal ** 2))
```

## 3. Chroma Features

### 3.1 Chroma Vector
Represents the distribution of energy across the 12 pitch classes.

```python
chroma = np.zeros(12)
for i in range(len(frequencies)):
    pitch_class = int(round(12 * np.log2(frequencies[i] / 440))) % 12
    chroma[pitch_class] += magnitudes[i]
```

## 4. MFCC (Mel-frequency cepstral coefficients)

### 4.1 Mel Filterbank
Converts frequencies to mel scale and applies triangular filters.

```python
def hz_to_mel(freq):
    return 2595 * np.log10(1 + freq / 700)

def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)

# Create mel filterbank
n_mels = 40
mel_filters = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
```

### 4.2 MFCC Calculation
```python
# Apply mel filterbank
mel_spectrum = np.dot(mel_filters, magnitudes)

# Take logarithm
log_mel_spectrum = np.log(mel_spectrum + 1e-10)

# Apply DCT
mfcc = scipy.fftpack.dct(log_mel_spectrum, type=2, norm='ortho')
```

## 5. Tempo and Beat Features

### 5.1 Tempo Estimation
```python
tempo, _ = librosa.beat.beat_track(y=signal, sr=sample_rate)
```

### 5.2 Beat Strength
```python
onset_env = librosa.onset.onset_strength(y=signal, sr=sample_rate)
beat_strength = np.mean(onset_env)
```

## 6. Onset Detection

### 6.1 Onset Strength
```python
onset_strength = librosa.onset.onset_strength(y=signal, sr=sample_rate)
```

### 6.2 Onset Rate
```python
onset_frames = librosa.onset.onset_detect(onset_envelope=onset_strength, sr=sample_rate)
onset_rate = len(onset_frames) / (len(signal) / sample_rate)
```

## 7. Harmonic Features

### 7.1 Harmonic-Percussive Separation
```python
harmonic, percussive = librosa.effects.hpss(signal)
```

### 7.2 Harmonic Ratio
```python
harmonic_ratio = np.sum(harmonic ** 2) / (np.sum(harmonic ** 2) + np.sum(percussive ** 2))
```

## 8. Dynamic Range

### 8.1 Dynamic Range Calculation
```python
dynamic_range = 20 * np.log10(np.max(np.abs(signal)) / (np.mean(np.abs(signal)) + 1e-10))
```

## 9. Spectral Contrast

### 9.1 Spectral Contrast Calculation
```python
spectral_contrast = np.std(magnitudes) / (np.mean(magnitudes) + 1e-10)
```

## 10. Spectral Flux

### 10.1 Spectral Flux Calculation
```python
spectral_flux = np.sum(np.diff(magnitudes) ** 2)
```

## Usage Notes

1. All spectral features are calculated using the Short-Time Fourier Transform (STFT)
2. Default parameters:
   - Sample rate: 22050 Hz
   - FFT size: 2048
   - Hop length: 512
   - Number of MFCCs: 13
   - Number of mel bands: 40

## Dependencies

- numpy
- librosa
- scipy
- soundfile