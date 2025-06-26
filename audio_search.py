import numpy as np
import os
from modified_feature_extraction import extract_features, get_feature_vectors, retrieve_features

def cosine_similarity(v1, v2):
    """
    Tính độ tương đồng cosine giữa hai vector
    
    Parameters:
    -----------
    v1, v2 : numpy.ndarray
        Hai vector đặc trưng cần tính toán độ tương đồng
        
    Returns:
    --------
    similarity : float
        Độ tương đồng cosine (giá trị từ -1 đến 1, càng cao càng giống nhau)
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0  # Xử lý trường hợp vector 0
        
    return dot_product / (norm_v1 * norm_v2)



def normalize_features(feature_vector):
    """
    Chuẩn hóa vector đặc trưng để đảm bảo các thành phần có thang đo tương đồng
    
    Parameters:
    -----------
    feature_vector : numpy.ndarray
        Vector đặc trưng cần chuẩn hóa
        
    Returns:
    --------
    normalized : numpy.ndarray
        Vector đặc trưng đã được chuẩn hóa
    """
    if np.all(feature_vector == 0):
        return feature_vector  # Tránh chia cho 0
        
    # Chuẩn hóa Min-Max để đưa giá trị về khoảng [0,1]
    min_val = np.min(feature_vector)
    max_val = np.max(feature_vector)
    
    if max_val == min_val:
        return np.zeros_like(feature_vector)
        
    return (feature_vector - min_val) / (max_val - min_val)

def search_similar_songs(input_audio, db_path, top_k=3):
    """
    Tìm kiếm những bài hát tương tự dựa trên đặc trưng âm thanh sử dụng Cosine similarity
    
    Parameters:
    -----------
    input_audio : str
        Đường dẫn đến file âm thanh cần tìm kiếm
    db_path : str
        Đường dẫn đến file cơ sở dữ liệu SQLite
    top_k : int, optional
        Số lượng kết quả trả về
        
    Returns:
    --------
    results : list
        Danh sách các bài hát tương tự nhất, mỗi phần tử là một dict với thông tin bài hát và similarity
    """
    # Trích xuất đặc trưng từ file âm thanh đầu vào
    input_features = extract_features(input_audio)
    if input_features is None:
        print(f"Không thể trích xuất đặc trưng từ file {input_audio}")
        return []
    
    # Lấy vector đặc trưng và chuẩn hóa
    input_vector = input_features['feature_vector']
    input_vector_norm = normalize_features(input_vector)
    
    # Lấy danh sách các vector đặc trưng từ cơ sở dữ liệu
    all_songs = get_feature_vectors(db_path)
    
    # Tính toán độ tương đồng với từng bài hát sử dụng Cosine similarity
    similarities = []
    for song in all_songs:
        song_vector = song['feature_vector']
        song_vector_norm = normalize_features(song_vector)
        
        # Tính cosine similarity - giá trị càng lớn càng giống nhau
        sim = cosine_similarity(input_vector_norm, song_vector_norm)
        similarities.append((song, sim))
    
    # Sắp xếp theo độ tương đồng giảm dần
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Chỉ lấy top_k kết quả
    top_results = similarities[:top_k]
    
    # Định dạng lại kết quả
    results = []
    for song, sim in top_results:
        results.append({
            'id': song['id'],
            'title': song['title'],
            'artist': song['artist'],
            'filename': song['filename'],
            'similarity': sim
        })
    
    return results

def main():
    DB_PATH = "./database/music_features_new.db"
    
    # Kiểm tra database tồn tại
    if not os.path.exists(DB_PATH):
        print(f"Không tìm thấy cơ sở dữ liệu {DB_PATH}")
        return
    
    # Lấy đường dẫn file âm thanh đầu vào
    input_audio = input("Nhập đường dẫn đến file âm thanh cần tìm kiếm: ")
    if not os.path.exists(input_audio):
        print(f"Không tìm thấy file {input_audio}")
        return
    
    print("\nĐang tìm kiếm bài hát tương tự...")
    
    # Tìm kiếm sử dụng Cosine similarity
    print("\n=== Kết quả tìm kiếm ===")
    results = search_similar_songs(input_audio, DB_PATH, top_k=3)
    for i, result in enumerate(results):
        print(f"{i+1}. {result['title']} (Ca sĩ: {result['artist']})")
        print(f"   Độ tương đồng: {result['similarity']:.4f}")

if __name__ == "__main__":
    main() 