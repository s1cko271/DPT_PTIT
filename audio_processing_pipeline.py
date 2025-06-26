import os
import time
import numpy as np
from modified_feature_extraction import extract_features
from audio_search import normalize_features, cosine_similarity

class AudioProcessingPipeline:
    """
    Pipeline xử lý âm thanh thống nhất:
    Nhận âm thanh → Trích xuất đặc trưng → So sánh → Trả kết quả
    """
    
    def __init__(self, database_path="./database/music_features_new.db"):
        """
        Khởi tạo pipeline xử lý âm thanh với Cosine similarity
        
        Parameters:
        -----------
        database_path : str
            Đường dẫn đến file cơ sở dữ liệu SQLite
        """
        self.database_path = database_path
        # Kiểm tra database
        if not os.path.exists(database_path):
            print(f"Cảnh báo: Không tìm thấy cơ sở dữ liệu tại {database_path}")

    def process(self, audio_file_path, top_k=3, verbose=True):
        """
        Xử lý một file âm thanh qua toàn bộ pipeline
        
        Parameters:
        -----------
        audio_file_path : str
            Đường dẫn đến file âm thanh cần xử lý
        top_k : int
            Số lượng kết quả trả về
        verbose : bool
            In thông tin chi tiết về quá trình xử lý
            
        Returns:
        --------
        dict
            Kết quả xử lý bao gồm:
            - input_info: Thông tin về file âm thanh đầu vào
            - features: Đặc trưng đã trích xuất
            - results: Danh sách các bài hát tương tự
            - processing_time: Thời gian xử lý từng bước
        """
        result = {
            'input_info': {
                'file_path': audio_file_path,
                'file_name': os.path.basename(audio_file_path)
            },
            'features': None,
            'results': [],
            'processing_time': {}
        }

        # BƯỚC 1: Kiểm tra file âm thanh đầu vào
        if not os.path.exists(audio_file_path):
            if verbose:
                print(f"Lỗi: Không tìm thấy file âm thanh tại {audio_file_path}")
            return result

        if verbose:
            print(f"\n[1. NHẬN ÂM THANH]")
            print(f"File âm thanh: {os.path.basename(audio_file_path)}")
        
        # BƯỚC 2: Trích xuất đặc trưng
        start_time = time.time()
        if verbose:
            print(f"\n[2. TRÍCH XUẤT ĐẶC TRƯNG]")
            print("Đang trích xuất đặc trưng âm thanh...")
            
        features = extract_features(audio_file_path)
        extraction_time = time.time() - start_time
        result['processing_time']['extraction'] = extraction_time
        
        if features is None:
            if verbose:
                print("Lỗi: Không thể trích xuất đặc trưng từ file âm thanh")
            return result
            
        result['features'] = features
        
        if verbose:
            print(f"Trích xuất đặc trưng hoàn tất ({extraction_time:.2f} giây)")
            print("Đặc trưng âm thanh:")
            for name, value in features.items():
                if name != 'feature_vector':
                    print(f"  - {name}: {value}")
        
        # BƯỚC 3: So sánh với cơ sở dữ liệu
        start_time = time.time()
        if verbose:
            print(f"\n[3. SO SÁNH ĐẶC TRƯNG]")
            print(f"Phương pháp so sánh: Cosine Similarity")
            print("Đang so sánh với cơ sở dữ liệu...")
        
        # Lấy vector đặc trưng từ dữ liệu đầu vào
        input_vector = features['feature_vector']
        input_vector_norm = normalize_features(input_vector)
        
        # Lấy danh sách các đặc trưng từ database
        from modified_feature_extraction import get_feature_vectors  # Import từ module mới
        all_songs = get_feature_vectors(self.database_path)
        
        if verbose:
            print(f"Tìm thấy {len(all_songs)} bản nhạc trong cơ sở dữ liệu")
            
        # Tính toán độ tương đồng với Cosine similarity
        similarities = []
        for song in all_songs:
            song_vector = song['feature_vector']
            song_vector_norm = normalize_features(song_vector)
            
            # Tính cosine similarity - giá trị cao hơn = giống hơn
            sim = cosine_similarity(input_vector_norm, song_vector_norm)
            similarities.append((song, sim))
        
        # Sắp xếp kết quả
        similarities.sort(key=lambda x: x[1], reverse=True)
        comparison_time = time.time() - start_time
        result['processing_time']['comparison'] = comparison_time
        
        if verbose:
            print(f"So sánh hoàn tất ({comparison_time:.2f} giây)")
        
        # BƯỚC 4: Trả kết quả
        if verbose:
            print(f"\n[4. TRẢ KẾT QUẢ]")
            
        # Lấy top_k kết quả
        top_results = similarities[:top_k]
        
        for i, (song, similarity) in enumerate(top_results):
            song_result = {
                'id': song['id'],
                'title': song['title'],
                'genre': song['genre'],
                'filename': song['filename'],
                'similarity': similarity
            }
            
            if verbose:
                print(f"{i+1}. {song['title']} (Thể loại: {song['genre']})")
                print(f"   Độ tương đồng: {similarity:.4f}")
            
            result['results'].append(song_result)
        
        # Tính tổng thời gian
        result['processing_time']['total'] = sum(result['processing_time'].values())
        
        if verbose:
            print(f"\nTổng thời gian xử lý: {result['processing_time']['total']:.2f} giây")
            
        return result


def run_pipeline(audio_path, db_path="./database/music_features_new.db", top_k=3):
    """
    Hàm tiện ích để chạy pipeline xử lý âm thanh với Cosine similarity
    
    Parameters:
    -----------
    audio_path : str
        Đường dẫn đến file âm thanh cần xử lý
    db_path : str
        Đường dẫn đến cơ sở dữ liệu
    top_k : int
        Số lượng kết quả trả về
        
    Returns:
    --------
    dict
        Kết quả xử lý từ pipeline
    """
    pipeline = AudioProcessingPipeline(database_path=db_path)
    return pipeline.process(audio_path, top_k=top_k)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline xử lý âm thanh")
    parser.add_argument("--audio", "-a", required=True, help="Đường dẫn đến file âm thanh")
    parser.add_argument("--db", default="./database/music_features_new.db", help="Đường dẫn đến cơ sở dữ liệu")
    parser.add_argument("--method", "-m", choices=["cosine", "euclidean"], default="cosine", 
                       help="Phương pháp tính độ tương đồng")
    parser.add_argument("--top", "-t", type=int, default=3, help="Số lượng kết quả trả về")
    
    args = parser.parse_args()
    
    # Chạy pipeline
    run_pipeline(args.audio, args.db, args.method, args.top) 