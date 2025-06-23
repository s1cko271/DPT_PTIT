import sqlite3
import joblib
import io
from tqdm import tqdm

def migrate_features(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Tạo bảng features nếu chưa có
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        song_id INTEGER,
        rms_energy REAL,
        zcr REAL,
        spectral_centroid REAL,
        spectral_bandwidth REAL,
        spectral_rolloff REAL,
        tempo REAL,
        feature_vector BLOB,
        FOREIGN KEY(song_id) REFERENCES songs(id)
    )
    ''')
    conn.commit()

    # Lấy tất cả bài hát và đặc trưng từ bảng songs
    cursor.execute("SELECT id, features FROM songs")
    all_songs = cursor.fetchall()

    print(f"Đang di chuyển đặc trưng của {len(all_songs)} bài hát sang bảng features...")
    for song_id, features_blob in tqdm(all_songs):
        features = joblib.load(io.BytesIO(features_blob))
        # Lưu feature_vector dưới dạng BLOB
        buffer = io.BytesIO()
        joblib.dump(features['feature_vector'], buffer)
        fv_blob = buffer.getvalue()
        cursor.execute('''
            INSERT INTO features (song_id, rms_energy, zcr, spectral_centroid, spectral_bandwidth, spectral_rolloff, tempo, feature_vector)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            song_id,
            float(features.get('rms_energy', 0)),
            float(features.get('zcr', 0)),
            float(features.get('spectral_centroid', 0)),
            float(features.get('spectral_bandwidth', 0)),
            float(features.get('spectral_rolloff', 0)),
            float(features.get('tempo', features.get('bpm', 0))),
            fv_blob
        ))
    conn.commit()
    print("Đã di chuyển xong đặc trưng sang bảng features!")
    conn.close()

if __name__ == "__main__":
    migrate_features('database/music_features.db') 