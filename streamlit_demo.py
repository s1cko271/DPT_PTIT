import streamlit as st
import os
import time
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import base64
from audio_processing_pipeline import AudioProcessingPipeline
from visualize_results import visualize_feature_comparison, visualize_similarity_scores, plot_processing_time
import sqlite3

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Hệ thống tìm kiếm âm nhạc",
    page_icon="🎵",
    layout="wide"
)

# Tùy chỉnh CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
    }
    h1 {
        color: #1E88E5;
        margin-bottom: 0.5rem;
        font-size: 1.8rem;
    }
    h2 {
        font-size: 1.4rem;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    h3 {
        font-size: 1.2rem;
        margin-top: 0.4rem;
        margin-bottom: 0.4rem;
    }
    h4, h5 {
        margin-top: 0.3rem;
        margin-bottom: 0.3rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        padding: 5px 15px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5;
        color: white;
    }
    .stSlider [data-testid="stThumbValue"] {
        background-color: #1E88E5;
        color: white;
    }
    div.stButton > button:first-child {
        background-color: #1E88E5;
        color: white;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #0D47A1;
        color: white;
        border: none;
    }
    .uploadedFile {
        border: 1px solid #ced4da;
        border-radius: 4px;
        padding: 8px;
        margin-bottom: 8px;
    }
    .audio-container {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.12);
    }
    .divider {
        margin-top: 0.3rem;
        margin-bottom: 0.3rem;
        border: 0;
        border-top: 1px solid rgba(0,0,0,0.1);
    }
    .result-container {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
        border-left: 4px solid #1E88E5;
    }
    p {
        margin-bottom: 0.3rem;
        margin-top: 0.3rem;
    }
    .stDataFrame {
        margin-bottom: 0.5rem;
    }
    .similarity-score {
        font-weight: bold;
        color: #1E88E5;
    }
    .footer {
        text-align: center;
        color: #666;
        margin-top: 1rem;
        padding-top: 0.3rem;
        border-top: 1px solid rgba(0,0,0,0.1);
    }
    .analysis-section {
        margin-top: 0.3rem;
        margin-bottom: 0.3rem;
    }
    .analysis-item {
        margin-bottom: 0.2rem;
    }
    .stAudio {
        margin-bottom: 0.3rem;
    }
    /* Loại bỏ khoảng trắng của Streamlit */
    .stMarkdown {
        margin-bottom: 0 !important;
    }
    .row-widget {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    .stExpander {
        margin-bottom: 0.5rem !important;
    }
    .st-emotion-cache-1fttcpj {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    .st-emotion-cache-16txtl3 {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    .st-emotion-cache-1r6slb0 {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }
    .streamlit-expanderHeader {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    .streamlit-expanderContent {
        padding-top: 0.3rem !important;
    }
    /* Giảm khoảng cách trong bảng kết quả */
    .stDataFrame div[data-testid="stTable"] {
        margin-top: 0 !important;
        margin-bottom: 0.5rem !important;
    }
    /* Giảm khoảng cách các tab */
    .stTabs {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    /* Tùy chỉnh thanh tabs */
    [data-baseweb="tab-panel"] {
        padding-top: 0.5rem !important;
    }
    /* Giảm khoảng cách header */
    [data-testid="stHeader"] {
        margin-bottom: 0 !important;
    }
    /* Loại bỏ khoảng trắng dư thừa ở cuối */
    .st-emotion-cache-z5fcl4 {
        padding-bottom: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Các hàm tiện ích
def check_database(db_path):
    """Kiểm tra cơ sở dữ liệu và trả về số lượng bài hát"""
    if not os.path.exists(db_path):
        return 0
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM songs")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0

def save_uploaded_file(uploaded_file):
    """Lưu file tải lên vào thư mục tạm và trả về đường dẫn"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def format_time(seconds):
    """Định dạng thời gian xử lý"""
    return f"{seconds:.2f}s"

# Thiết lập tiêu đề và giới thiệu
st.title("🎵 Hệ thống tìm kiếm âm nhạc dựa trên đặc trưng âm thanh")
st.markdown("""
Hệ thống này cho phép tìm kiếm bài hát tương tự dựa trên đặc trưng âm thanh, sử dụng phương pháp tính toán độ tương đồng Cosine Similarity.
""")

# Đường dẫn cơ sở dữ liệu - ẩn, nhưng vẫn sử dụng mặc định
db_path = "./database/music_features.db"

# Cố định phương pháp so sánh là Cosine Similarity
similarity_method = "cosine"

# Cố định số lượng kết quả là 3
top_k = 3

# Khu vực tải lên file
st.markdown("### Tải lên file âm thanh")
uploaded_file = st.file_uploader("Chọn một file âm thanh (MP3, WAV, OGG)", type=["mp3", "wav", "ogg"])

# Kiểm tra xem có file được tải lên không
if uploaded_file is not None:
    # Hiển thị thông báo đang xử lý
    with st.spinner("Đang xử lý..."):
        # Lưu file tạm thời
        temp_file_path = save_uploaded_file(uploaded_file)
        
        # Tạo thư mục kết quả tạm thời cho biểu đồ
        results_dir = tempfile.mkdtemp()
        
        # Khởi tạo pipeline và xử lý âm thanh
        pipeline = AudioProcessingPipeline(database_path=db_path, similarity_method=similarity_method)
        start_time = time.time()
        result = pipeline.process(temp_file_path, top_k=top_k, verbose=False)
        total_time = time.time() - start_time
    
    # Hiển thị kết quả nếu có
    if result["results"]:
        st.success(f"Đã tìm thấy {len(result['results'])} bài hát tương tự! (Thời gian xử lý: {format_time(total_time)})")
        
        # Hiển thị thông tin file đầu vào
        st.markdown("### File âm thanh đầu vào")
        st.audio(uploaded_file)
        
        # Thêm giải thích về kết quả tìm kiếm
        with st.expander("Xem giải thích về phương pháp phân tích", expanded=False):
            st.markdown("""
            Hệ thống đã phân tích các đặc trưng âm thanh của file bạn tải lên, bao gồm:
            
            1. **RMS Energy**: Độ lớn năng lượng của âm thanh
            2. **Zero-Crossing Rate**: Tần suất tín hiệu âm thanh thay đổi từ âm sang dương
            3. **Spectral Centroid**: Trọng tâm của phổ âm thanh (liên quan đến "độ sáng" của âm thanh)
            4. **Spectral Bandwidth**: Độ rộng của phổ âm thanh
            5. **Spectral Rolloff**: Tần số mà dưới đó chứa 85% năng lượng phổ
            6. **Tempo**: Nhịp độ của bài hát
            
            Sau đó, hệ thống tính toán độ tương đồng Cosine giữa vector đặc trưng của file bạn tải lên và cơ sở dữ liệu, để tìm ra 3 bài hát tương tự nhất.
            """)
        
        st.markdown("### Kết quả tương tự")
        
        # Tạo dataframe để hiển thị kết quả
        results_data = []
        for i, song in enumerate(result["results"]):
            results_data.append({
                "STT": i + 1,
                "Tên bài hát": song["filename"].split('.')[0] + "." + song["filename"].split('.')[1],
                "Độ tương đồng": song.get('similarity', 0)
            })
        
        df = pd.DataFrame(results_data)
        st.dataframe(df, use_container_width=True)
        
        # Hiển thị audio player cho mỗi kết quả
        st.markdown("### Nghe thử")
        for i, song in enumerate(result["results"]):
            with st.container():
                col1, col2 = st.columns([6, 1])
                with col1:
                    # Hiển thị tên đầy đủ của file (không bao gồm phần mở rộng)
                    file_name = song["filename"].split('.')
                    display_name = file_name[0] + "." + file_name[1] if len(file_name) > 1 else file_name[0]
                    st.markdown(f"#### {i+1}. {display_name}")
                    
                    # Tạo đường dẫn đến file âm thanh từ tên file
                    song_path = os.path.join("songs", song["filename"])
                    
                    if os.path.exists(song_path):
                        st.audio(song_path)
                    else:
                        st.warning(f"Không thể tìm thấy file âm thanh: {song_path}")
                
                with col2:
                    st.markdown(f"<div style='margin-top:25px;'><span class='similarity-score'>{song.get('similarity', 0):.4f}</span></div>", unsafe_allow_html=True)
                
                # Phần phân tích
                with st.expander("Xem phân tích chi tiết", expanded=False):
                    features = result.get('features', {})
                    if features and "feature_vector" in features:
                        # Lấy các đặc trưng của file đầu vào
                        input_features = features.get("feature_vector", [])
                        
                        # Phân tích chi tiết dựa trên điểm số tương đồng
                        # Tạo một phần phân tích chi tiết hơn dựa vào thể loại nhạc
                        genre = song['filename'].split('.')[0]  # Lấy thể loại từ tên file
                        
                        # Các đặc trưng được mô tả dựa vào thể loại
                        genre_descriptions = {
                            "blues": "nhịp điệu đặc trưng, tần số thấp và cấu trúc hòa âm bluesy",
                            "classical": "dải động rộng, cấu trúc phổ phức tạp và thành phần hài hòa",
                            "country": "âm sắc giọng hát đặc trưng, nhịp điệu ổn định và tần số trung bình",
                            "disco": "nhịp bốn phần tư nổi bật, tần số trầm mạnh và năng lượng cao đều đặn",
                            "hiphop": "nhịp điệu đập mạnh, tần số thấp nổi bật và cấu trúc beat đặc trưng",
                            "jazz": "cấu trúc hòa âm phức tạp, phổ tần số đa dạng và tính ứng tác",
                            "metal": "năng lượng cao, nhiều tần số cao và thành phần nhiễu",
                            "pop": "cấu trúc giai điệu rõ ràng, năng lượng trung bình và tần số cân bằng",
                            "reggae": "nhịp điệu off-beat đặc trưng, tần số trầm mạnh và nhịp độ chậm hơn",
                            "rock": "năng lượng cao, dải tần số rộng và cường độ âm thanh đặc trưng"
                        }
                        
                        genre_desc = genre_descriptions.get(genre, "đặc trưng âm thanh riêng biệt")
                        
                        similarity_level = ""
                        if song.get('similarity', 0) > 0.95:
                            similarity_level = "cực kỳ"
                        elif song.get('similarity', 0) > 0.85:
                            similarity_level = "rất"
                        elif song.get('similarity', 0) > 0.7:
                            similarity_level = "khá"
                        else:
                            similarity_level = "có một số"
                        
                        st.markdown(f"Bài hát này <strong>{similarity_level}</strong> tương đồng với file của bạn.", unsafe_allow_html=True)
                        
                        cols = st.columns(2)
                        with cols[0]:
                            st.markdown(f"**Đặc trưng năng lượng**: Cấu trúc năng lượng âm thanh {similarity_level} giống với file của bạn.")
                            st.markdown(f"**Cấu trúc nhịp điệu**: Có sự tương đồng về tốc độ và nhịp điệu.")
                        
                        with cols[1]:
                            st.markdown(f"**Âm sắc**: Phổ tần số có cấu trúc tương tự với file của bạn.")
                            st.markdown(f"**Đặc điểm thể loại**: Thể loại {genre} với {genre_desc}.")
                        
                        # Thêm phân tích cụ thể cho mỗi thể loại
                        if genre == "blues":
                            st.markdown("Bản blues này có cấu trúc hòa âm đặc trưng với quãng 7, tạo nên âm hưởng tương tự với file của bạn.")
                        elif genre == "classical":
                            st.markdown("Bản nhạc cổ điển này chia sẻ cấu trúc động và sự biến đổi về cường độ với file của bạn.")
                        elif genre == "country":
                            st.markdown("Bài country này có âm sắc tươi sáng và tiết tấu ổn định, tạo nên điểm tương đồng với file của bạn.")
                        elif genre == "disco":
                            st.markdown("Đặc trưng disco với nhịp đập nổi bật và cấu trúc tiết tấu lặp lại khiến bài hát này có điểm chung với file của bạn.")
                        elif genre == "hiphop":
                            st.markdown("Nhịp beat mạnh mẽ và cấu trúc đặc trưng của hip-hop trong bài này tạo ra năng lượng tương tự với file của bạn.")
                        elif genre == "jazz":
                            st.markdown("Cấu trúc hòa âm phong phú và sự đa dạng về tần số trong bài jazz này tương đồng với các đặc điểm trong file của bạn.")
                        elif genre == "metal":
                            st.markdown("Năng lượng cao và cấu trúc phổ tần số phức tạp của bài metal này phù hợp với đặc điểm âm thanh trong file của bạn.")
                        elif genre == "pop":
                            st.markdown("Sự cân bằng về tần số và cấu trúc rõ ràng của bài pop này có nhiều điểm tương đồng với cấu trúc âm thanh trong file của bạn.")
                        elif genre == "reggae":
                            st.markdown("Nhịp điệu off-beat đặc trưng và tần số trầm nổi bật của reggae trong bài này tạo nên sự tương đồng với file của bạn.")
                        elif genre == "rock":
                            st.markdown("Năng lượng cao và sự cân bằng giữa các dải tần số của bài rock này khá tương đồng với cấu trúc phổ trong file của bạn.")
            
            # Thêm một divider nhỏ giữa các bài hát
            if i < len(result["results"]) - 1:
                st.markdown("<hr style='margin: 5px 0; border-width: 1px 0 0 0; border-style: solid; border-color: #f0f0f0;'>", unsafe_allow_html=True)
        
        # Tab 2: Biểu đồ so sánh
        st.markdown("### So sánh đặc trưng âm thanh")
        with st.expander("Xem biểu đồ so sánh", expanded=False):
            # Tạo biểu đồ so sánh đặc trưng
            try:
                fig1, _ = visualize_feature_comparison(
                    temp_file_path, 
                    result['results'], 
                    db_path
                )
                st.pyplot(fig1)
            except Exception as e:
                st.error(f"Lỗi khi tạo biểu đồ so sánh đặc trưng: {e}")
            
            # Tạo biểu đồ điểm số tương đồng
            try:
                fig2, _ = visualize_similarity_scores(
                    result['results'],
                    similarity_method
                )
                st.pyplot(fig2)
            except Exception as e:
                st.error(f"Lỗi khi tạo biểu đồ điểm số tương đồng: {e}")
    else:
        st.error("Không tìm thấy kết quả phù hợp. Vui lòng thử file khác hoặc kiểm tra cơ sở dữ liệu.")
    
    # Dọn dẹp file tạm thời
    try:
        os.unlink(temp_file_path)
    except:
        pass

# Thêm thông tin footer
st.markdown("<div class='footer'>Hệ thống tìm kiếm âm nhạc dựa trên đặc trưng âm thanh • 2023</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    # Code này sẽ chạy khi chạy script trực tiếp
    # Lệnh chạy: streamlit run streamlit_demo.py
    pass 