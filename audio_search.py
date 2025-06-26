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

def euclidean_distance(v1, v2):
    """
    Tính khoảng cách Euclidean giữa hai vector
    
    Parameters:
    -----------
    v1, v2 : numpy.ndarray
        Hai vector đặc trưng cần tính toán khoảng cách
        
    Returns:
    --------
    distance : float
        Khoảng cách Euclidean (giá trị từ 0 trở lên, càng thấp càng giống nhau)
    """
    return np.linalg.norm(v1 - v2)

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

def search_similar_songs(input_audio, db_path, metric='cosine', top_k=3):
    """
    Tìm kiếm những bài hát tương tự dựa trên đặc trưng âm thanh
    
    Parameters:
    -----------
    input_audio : str
        Đường dẫn đến file âm thanh cần tìm kiếm
    db_path : str
        Đường dẫn đến file cơ sở dữ liệu SQLite
    metric : str, optional
        Phương pháp tính độ tương đồng ('cosine' hoặc 'euclidean')
    top_k : int, optional
        Số lượng kết quả trả về
        
    Returns:
    --------
    results : list
        Danh sách các bài hát tương tự nhất, mỗi phần tử là một tuple (song_info, similarity)
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
    
    # Tính toán độ tương đồng với từng bài hát
    similarities = []
    for song in all_songs:
        song_vector = song['feature_vector']
        song_vector_norm = normalize_features(song_vector)
        
        if metric.lower() == 'cosine':
            # Với cosine similarity, giá trị càng lớn càng giống nhau
            sim = cosine_similarity(input_vector_norm, song_vector_norm)
            similarities.append((song, sim))
        else:  # euclidean
            # Với euclidean distance, giá trị càng nhỏ càng giống nhau
            dist = euclidean_distance(input_vector_norm, song_vector_norm)
            similarities.append((song, -dist))  # Đổi dấu để sắp xếp giảm dần
    
    # Sắp xếp theo độ tương đồng giảm dần
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Chỉ lấy top_k kết quả
    top_results = similarities[:top_k]
    
    # Định dạng lại kết quả
    results = []
    for song, sim in top_results:
        if metric.lower() == 'cosine':
            # Giữ nguyên giá trị cosine
            results.append({
                'id': song['id'],
                'title': song['title'],
                'artist': song['artist'],
                'filename': song['filename'],
                'similarity': sim
            })
        else:  # euclidean
            # Chuyển lại thành giá trị dương cho khoảng cách
            results.append({
                'id': song['id'],
                'title': song['title'],
                'artist': song['artist'],
                'filename': song['filename'],
                'distance': -sim  # Đổi dấu lại để hiển thị khoảng cách thực
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
    print("\n=== Kết quả sử dụng Cosine similarity ===")
    cosine_results = search_similar_songs(input_audio, DB_PATH, metric='cosine', top_k=3)
    for i, result in enumerate(cosine_results):
        print(f"{i+1}. {result['title']} (Ca sĩ: {result['artist']})")
        print(f"   Độ tương đồng: {result['similarity']:.4f}")
    
    # Tìm kiếm sử dụng Euclidean distance
    print("\n=== Kết quả sử dụng Euclidean distance ===")
    euclidean_results = search_similar_songs(input_audio, DB_PATH, metric='euclidean', top_k=3)
    for i, result in enumerate(euclidean_results):
        print(f"{i+1}. {result['title']} (Ca sĩ: {result['artist']})")
        print(f"   Khoảng cách: {result['distance']:.4f}")

if __name__ == "__main__":
    main() 