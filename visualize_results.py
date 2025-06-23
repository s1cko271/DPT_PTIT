import matplotlib.pyplot as plt
import numpy as np
import os
from audio_processing_pipeline import run_pipeline
from modified_feature_extraction import extract_features

def visualize_feature_comparison(input_audio, similar_songs, db_path="./music_features.db", 
                               feature_names=None, title=None):
    """
    Trực quan hóa so sánh đặc trưng giữa file âm thanh đầu vào và các bài hát tương tự
    
    Parameters:
    -----------
    input_audio : str
        Đường dẫn đến file âm thanh đầu vào
    similar_songs : list
        Danh sách các bài hát tương tự từ kết quả pipeline
    db_path : str
        Đường dẫn đến cơ sở dữ liệu
    feature_names : list
        Danh sách tên các đặc trưng cần trực quan hóa
    title : str
        Tiêu đề của biểu đồ
    """
    from audio_search import retrieve_features
    
    # Trích xuất đặc trưng từ file đầu vào
    input_features = extract_features(input_audio)
    if input_features is None:
        print(f"Lỗi: Không thể trích xuất đặc trưng từ file {input_audio}")
        return
    
    # Các đặc trưng mặc định nếu không chỉ định
    if feature_names is None:
        feature_names = [
            'rms_energy', 'zcr', 'spectral_centroid', 
            'spectral_bandwidth', 'spectral_rolloff', 'tempo'
        ]
    
    # Chuẩn bị dữ liệu cho biểu đồ
    feature_values = [input_features[name] for name in feature_names]
    similar_features = []
    
    for song in similar_songs:
        song_id = song.get('id')
        if song_id:
            song_info = retrieve_features(db_path, song_id=song_id)
            if song_info:
                values = [song_info['features'][name] for name in feature_names]
                similar_features.append((song['title'], values))
    
    # Chuẩn hóa giá trị đặc trưng để dễ so sánh
    max_values = []
    for i in range(len(feature_names)):
        values = [feature_values[i]] + [f[1][i] for f in similar_features]
        max_val = max(values)
        max_values.append(max_val if max_val != 0 else 1)
    
    normalized_input = [feature_values[i]/max_values[i] for i in range(len(feature_names))]
    normalized_similar = []
    
    for title, values in similar_features:
        normalized = [values[i]/max_values[i] for i in range(len(feature_names))]
        normalized_similar.append((title, normalized))
    
    # Vẽ biểu đồ radar (spider/polar)
    angles = np.linspace(0, 2*np.pi, len(feature_names), endpoint=False).tolist()
    # Khép vòng tròn
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # Vẽ cho file đầu vào
    values = normalized_input + [normalized_input[0]]  # Khép vòng tròn
    ax.plot(angles, values, 'o-', linewidth=2, label=os.path.basename(input_audio))
    ax.fill(angles, values, alpha=0.1)
    
    # Vẽ cho các bài hát tương tự
    markers = ['s', '^', 'D', 'v', '<']  # Các marker khác nhau cho mỗi bài hát
    for i, (song_title, values) in enumerate(normalized_similar):
        values_loop = values + [values[0]]  # Khép vòng tròn
        marker = markers[i % len(markers)]
        ax.plot(angles, values_loop, marker=marker, linestyle='-', linewidth=1.5, 
                label=f"{i+1}. {song_title}")
        ax.fill(angles, values_loop, alpha=0.05)
    
    # Thêm các nhãn cho các đặc trưng
    plt.xticks(angles[:-1], feature_names)
    
    # Thêm tiêu đề và chú thích
    if title:
        plt.title(title, size=15, y=1.1)
    else:
        plt.title('So sánh đặc trưng âm thanh', size=15, y=1.1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    return fig, ax

def visualize_similarity_scores(similar_songs, similarity_method="cosine", title=None):
    """
    Trực quan hóa điểm số tương đồng của các bài hát tương tự
    
    Parameters:
    -----------
    similar_songs : list
        Danh sách các bài hát tương tự từ kết quả pipeline
    similarity_method : str
        Phương pháp tính toán độ tương đồng đã sử dụng
    title : str
        Tiêu đề của biểu đồ
    """
    # Chuẩn bị dữ liệu
    song_names = []
    scores = []
    
    for song in similar_songs:
        song_names.append(f"{song['title']}\n({song['artist']})")
        if similarity_method.lower() == 'cosine':
            scores.append(song.get('similarity', 0))
        else:  # euclidean
            scores.append(song.get('distance', 0))
    
    # Sắp xếp dữ liệu theo điểm số
    if similarity_method.lower() == 'cosine':
        # Cao đến thấp cho cosine
        sorted_indices = np.argsort(scores)[::-1]  
    else:
        # Thấp đến cao cho euclidean
        sorted_indices = np.argsort(scores)
        
    song_names = [song_names[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]
    
    # Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        range(len(song_names)), 
        scores, 
        color='skyblue', 
        edgecolor='navy'
    )
    
    # Thêm giá trị lên thanh
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height*1.01, 
            f'{scores[i]:.4f}', 
            ha='center', 
            va='bottom',
            fontweight='bold'
        )
    
    # Thêm nhãn và tiêu đề
    plt.xticks(range(len(song_names)), song_names, rotation=20, ha='right')
    
    if similarity_method.lower() == 'cosine':
        plt.ylabel('Cosine Similarity (cao hơn = giống nhau hơn)')
        if not title:
            title = 'Độ tương đồng Cosine với các bài hát tương tự'
    else:  # euclidean
        plt.ylabel('Euclidean Distance (thấp hơn = giống nhau hơn)')
        if not title:
            title = 'Khoảng cách Euclidean đến các bài hát tương tự'
    
    plt.title(title)
    plt.tight_layout()
    
    return fig, ax

def plot_processing_time(processing_times):
    """
    Trực quan hóa thời gian xử lý từng bước trong pipeline
    
    Parameters:
    -----------
    processing_times : dict
        Dictionary chứa thời gian xử lý từng bước
    """
    # Chuẩn bị dữ liệu
    steps = list(processing_times.keys())
    if 'total' in steps:
        steps.remove('total')  # Loại bỏ tổng thời gian
    times = [processing_times[step] for step in steps]
    
    # Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(steps, times, color='lightgreen', edgecolor='darkgreen')
    
    # Thêm giá trị lên thanh
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(
            width*1.01, 
            bar.get_y() + bar.get_height()/2., 
            f'{times[i]:.2f}s', 
            ha='left', 
            va='center',
            fontweight='bold'
        )
    
    # Thêm nhãn và tiêu đề
    plt.xlabel('Thời gian (giây)')
    plt.title('Thời gian xử lý theo từng bước')
    plt.tight_layout()
    
    return fig, ax

def visualize_pipeline_results(input_audio_path, database_path="./music_features.db", 
                             similarity_method="cosine", top_k=3, save_path=None):
    """
    Tạo báo cáo trực quan cho toàn bộ pipeline xử lý âm thanh
    
    Parameters:
    -----------
    input_audio_path : str
        Đường dẫn đến file âm thanh đầu vào
    database_path : str
        Đường dẫn đến cơ sở dữ liệu
    similarity_method : str
        Phương pháp tính toán độ tương đồng
    top_k : int
        Số lượng kết quả trả về
    save_path : str
        Đường dẫn để lưu các biểu đồ (không lưu nếu là None)
    """
    # Chạy pipeline
    print(f"Đang xử lý file âm thanh: {input_audio_path}")
    pipeline_result = run_pipeline(
        input_audio_path, 
        database_path, 
        similarity_method, 
        top_k
    )
    
    # Nếu không có kết quả, thoát
    if not pipeline_result['results']:
        print("Không có kết quả từ pipeline")
        return
    
    # Tạo thư mục lưu nếu cần
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    # 1. Biểu đồ so sánh đặc trưng
    fig1, _ = visualize_feature_comparison(
        input_audio_path, 
        pipeline_result['results'], 
        database_path
    )
    
    if save_path:
        fig1.savefig(os.path.join(save_path, "feature_comparison.png"), 
                     bbox_inches="tight", dpi=300)
    
    # 2. Biểu đồ điểm số tương đồng
    fig2, _ = visualize_similarity_scores(
        pipeline_result['results'],
        similarity_method
    )
    
    if save_path:
        fig2.savefig(os.path.join(save_path, f"{similarity_method}_scores.png"), 
                     bbox_inches="tight", dpi=300)
    
    # 3. Biểu đồ thời gian xử lý
    fig3, _ = plot_processing_time(pipeline_result['processing_time'])
    
    if save_path:
        fig3.savefig(os.path.join(save_path, "processing_time.png"), 
                     bbox_inches="tight", dpi=300)
    
    # Hiển thị tất cả biểu đồ
    plt.show()
    
    return pipeline_result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Trực quan hóa kết quả tìm kiếm âm nhạc")
    parser.add_argument("--audio", "-a", required=True, help="Đường dẫn đến file âm thanh")
    parser.add_argument("--db", default="./music_features.db", help="Đường dẫn đến cơ sở dữ liệu")
    parser.add_argument("--method", "-m", choices=["cosine", "euclidean"], default="cosine", 
                      help="Phương pháp tính độ tương đồng")
    parser.add_argument("--top", "-t", type=int, default=3, help="Số lượng kết quả trả về")
    parser.add_argument("--save", "-s", help="Thư mục để lưu biểu đồ")
    
    args = parser.parse_args()
    
    # Chạy trực quan hóa
    visualize_pipeline_results(args.audio, args.db, args.method, args.top, args.save) 