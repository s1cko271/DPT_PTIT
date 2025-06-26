import streamlit as st
import os
import time
import tempfile
from audio_processing_pipeline import AudioProcessingPipeline

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Tìm kiếm nhạc tương tự",
    page_icon="🎵",
    layout="wide"
)

# CSS tối ưu - compact và đẹp mắt
st.markdown("""
<style>
    /* Ẩn menu và footer của Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .css-1rs6os {visibility: hidden;}
    .css-17eq0hr {visibility: hidden;}
    
    /* Ẩn sidebar toggle */
    .css-1d391kg {display: none;}
    
    /* Loại bỏ padding dư thừa */
    .main .block-container {
        padding: 1rem 1rem 0rem;
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Header compact */
    h1 {
        color: #1E88E5;
        text-align: center;
        margin: 0 0 2rem 0;
        font-size: 2.2rem;
        font-weight: 600;
    }
    
    h3 {
        margin: 1rem 0 0.5rem 0;
        font-size: 1.3rem;
    }
    
    /* File uploader đơn giản */
    .stFileUploader {
        margin: 1rem 0;
        padding: 1rem;
        border: 2px dashed #1E88E5;
        border-radius: 10px;
        background: #f8f9fa;
    }
    
    /* Result items compact và đẹp */
    .result-item {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Similarity score đẹp hơn */
    .similarity-score {
        background: #1E88E5;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        text-align: center;
    }
    
    /* Audio player compact */
    .stAudio {
        margin: 0.5rem 0;
    }
    
    /* Success message đơn giản */
    .stSuccess {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Info message */
    .stInfo {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Error message */
    .stError {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Loại bỏ khoảng trắng dư thừa */
    .element-container {
        margin-bottom: 0.5rem !important;
    }
    
    /* Footer compact */
    hr {
        margin: 2rem 0 1rem 0;
        border-color: #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

def save_uploaded_file(uploaded_file):
    """Lưu file tải lên vào thư mục tạm và trả về đường dẫn"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

# Tiêu đề
st.title("🎵 Tìm kiếm nhạc tương tự")

# Khu vực upload đơn giản
st.markdown("### 📁 Tải lên file nhạc của bạn")
uploaded_file = st.file_uploader("Chọn file âm thanh (MP3, WAV, OGG)", type=["mp3", "wav", "ogg"])

# Xử lý khi có file upload
if uploaded_file is not None:
    # Hiển thị file đã upload trong container compact
    with st.container():
        st.markdown("### 🎧 File của bạn:")
        st.audio(uploaded_file)
    
    # Xử lý tìm kiếm
    with st.spinner("Đang tìm kiếm các bài hát tương tự..."):
        # Lưu file tạm
        temp_file_path = save_uploaded_file(uploaded_file)
        
        # Khởi tạo pipeline với database mới
        db_path = "./database/music_features_new.db"
        pipeline = AudioProcessingPipeline(database_path=db_path, similarity_method="cosine")
        
        # Xử lý và tìm kiếm
        start_time = time.time()
        result = pipeline.process(temp_file_path, top_k=3, verbose=False)
        processing_time = time.time() - start_time
    
    # Hiển thị kết quả trong container
    if result["results"]:
        st.success(f"✅ Tìm thấy 3 bài hát tương tự! (Thời gian: {processing_time:.2f}s)")
        
        st.markdown("### 🎵 Các bài hát tương tự:")
        
        # Hiển thị từng bài hát trong container compact
        for i, song in enumerate(result["results"]):
            with st.container():
                st.markdown(f'<div class="result-item">', unsafe_allow_html=True)
                
                # Tên bài hát và điểm số
                col1, col2 = st.columns([3, 1])
                with col1:
                    file_name = song["filename"].replace('.wav', '').replace('.', ' - ')
                    st.markdown(f"**{i+1}. {file_name}**")
                with col2:
                    similarity = song.get('similarity', 0)
                    st.markdown(f'<div class="similarity-score">{similarity:.1%}</div>', unsafe_allow_html=True)
                
                # Audio player
                song_path = os.path.join("songs", song["filename"])
                if os.path.exists(song_path):
                    st.audio(song_path)
                else:
                    st.error(f"Không tìm thấy file: {song['filename']}")
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.error("❌ Không tìm thấy bài hát tương tự. Vui lòng thử file khác.")
    
    # Dọn dẹp file tạm
    try:
        os.unlink(temp_file_path)
    except:
        pass

else:
    st.info("👆 Hãy tải lên một file nhạc để bắt đầu tìm kiếm!")

# Footer đơn giản
st.markdown("---")
st.markdown("💡 **Hướng dẫn:** Tải lên file nhạc và hệ thống sẽ tìm 3 bài hát tương tự nhất")

if __name__ == "__main__":
    # Code này sẽ chạy khi chạy script trực tiếp
    # Lệnh chạy: streamlit run streamlit_demo.py
    pass 