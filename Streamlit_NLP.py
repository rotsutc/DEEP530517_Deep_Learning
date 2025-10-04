import streamlit as st
import spacy
import pandas as pd
import plotly.express as px
from collections import Counter
import re

# Page configuration
st.set_page_config(
    page_title="Ứng Dụng Phân Tích Văn Bản",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Hide Streamlit branding and menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    div[data-testid="stToolbar"] {display:none;}
    .stApp > header {display:none;}
    
    /* Global Styles */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Main Header with Animation */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        color: #667eea;
        margin-bottom: 1rem;
        text-shadow: 0 2px 4px rgba(102, 126, 234, 0.2);
        animation: fadeInUp 1s ease-out;
        position: relative;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    
    .main-header::after {
        content: '';
        position: absolute;
        bottom: -10px;
        left: 50%;
        transform: translateX(-50%);
        width: 100px;
        height: 4px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 2px;
        opacity: 0.8;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Sub Header */
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        font-weight: 500;
        text-align: center;
        color: #475569;
        margin-bottom: 3rem;
        animation: fadeInUp 1s ease-out 0.2s both;
        line-height: 1.4;
        letter-spacing: 0.01em;
    }
    
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    .sidebar-section {
        background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .sidebar-section:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2.5rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
        width: 100% !important;
        min-width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Sidebar Button Specific Styling */
    .sidebar .stButton {
        width: 100% !important;
    }
    
    .sidebar .stButton > div {
        width: 100% !important;
    }
    
    .sidebar .stButton > div > button {
        width: 100% !important;
        min-width: 100% !important;
        max-width: 100% !important;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f8fafc;
        padding: 8px;
        border-radius: 15px;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
        display: flex;
        justify-content: space-between;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 70px;
        white-space: nowrap;
        background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        border: 2px solid transparent;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        flex: 1;
        min-width: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 0 8px;
        color: #475569;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        color: #667eea !important;
        border: 2px solid #667eea;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: 2px solid #5a67d8;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transform: translateY(-1px);
    }
    
    /* Responsive tab text */
    @media (max-width: 1200px) {
        .stTabs [data-baseweb="tab"] {
            font-size: 1rem;
            height: 65px;
        }
    }
    
    @media (max-width: 768px) {
        .stTabs [data-baseweb="tab"] {
            font-size: 0.9rem;
            height: 60px;
            padding: 0 4px;
        }
    }
    
    @media (max-width: 480px) {
        .stTabs [data-baseweb="tab"] {
            font-size: 0.8rem;
            height: 55px;
            padding: 0 2px;
        }
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.3);
    }
    
    /* Dataframe Styling */
    .dataframe {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: none;
    }
    
    .dataframe th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 1rem;
    }
    
    .dataframe td {
        padding: 0.75rem 1rem;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .dataframe tr:nth-child(even) {
        background-color: #f8fafc;
    }
    
    .dataframe tr:hover {
        background-color: #e2e8f0;
        transition: background-color 0.3s ease;
    }
    
    /* Entity Highlighting */
    .entity-highlight {
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        margin: 0 2px;
        display: inline-block;
        transition: transform 0.2s ease;
    }
    
    .entity-highlight:hover {
        transform: scale(1.05);
    }
    
    /* Info Boxes */
    .highlight-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #ff6b6b;
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.2);
    }
    
    .success-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #4ecdc4;
        box-shadow: 0 8px 25px rgba(78, 205, 196, 0.2);
    }
    
    /* Plotly Chart Styling */
    .js-plotly-plot {
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        overflow: hidden;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #64748b;
        font-size: 1.1rem;
        font-weight: 500;
    }
    
    /* Loading Animation */
    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        
        .sub-header {
            font-size: 1.1rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            font-size: 0.9rem;
        }
    }
    
    /* Text Area Styling */
    .stTextArea > div > div > textarea {
        border: 2px solid #e2e8f0;
        border-radius: 15px;
        padding: 1rem;
        font-size: 1rem;
        line-height: 1.6;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        resize: vertical;
        min-height: 120px;
    }
    
    .stTextArea > div > div > textarea:focus {
        border: 2px solid #667eea;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
        outline: none;
        background: #ffffff;
    }
    
    .stTextArea > div > div > textarea:hover {
        border: 2px solid #cbd5e1;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    }
    
    /* Selectbox Styling */
    .stSelectbox > div > div {
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover {
        border: 2px solid #cbd5e1;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    }
    
    .stSelectbox > div > div:focus-within {
        border: 2px solid #667eea;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }
</style>
""", unsafe_allow_html=True)

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    try:
        # Try to load the English model
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        st.error("""
        **Không tìm thấy mô hình tiếng Anh của spaCy!** 
        
        Vui lòng cài đặt bằng lệnh:
        ```
        python -m spacy download en_core_web_sm
        ```
        """)
        return None

def highlight_entities(text, entities):
    """Highlight named entities in text with different colors"""
    if not entities:
        return text
    
    # Color mapping for different entity types
    colors = {
        'PERSON': '#FF6B6B',      # Red
        'ORG': '#4ECDC4',         # Teal
        'GPE': '#45B7D1',         # Blue
        'LOC': '#96CEB4',         # Green
        'DATE': '#FFEAA7',        # Yellow
        'TIME': '#DDA0DD',        # Plum
        'MONEY': '#98D8C8',       # Mint
        'PERCENT': '#F7DC6F',     # Light Yellow
        'CARDINAL': '#BB8FCE',    # Light Purple
        'ORDINAL': '#85C1E9',     # Light Blue
        'QUANTITY': '#F8C471',    # Orange
        'EVENT': '#F1948A',       # Light Red
        'FAC': '#82E0AA',         # Light Green
        'LANGUAGE': '#F9E79F',    # Light Yellow
        'LAW': '#D5DBDB',         # Light Gray
        'NORP': '#AED6F1',        # Light Blue
        'PRODUCT': '#D7BDE2',     # Light Purple
        'WORK_OF_ART': '#A9DFBF', # Light Green
    }
    
    highlighted_text = text
    offset = 0
    
    # Sort entities by start position (descending) to avoid offset issues
    sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
    
    for entity in sorted_entities:
        start = entity['start'] + offset
        end = entity['end'] + offset
        label = entity['label']
        color = colors.get(label, '#E0E0E0')  # Default gray color
        
        # Create highlighted span
        highlighted_span = f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px; font-weight: bold;">{entity["text"]} ({label})</span>'
        
        # Replace the text
        highlighted_text = highlighted_text[:start] + highlighted_span + highlighted_text[end:]
        offset += len(highlighted_span) - (end - start)
    
    return highlighted_text

def main():
    # Main header with enhanced styling
    st.markdown('<h1 class="main-header">📝 Ứng Dụng Phân Tích Văn Bản</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Phân tích văn bản bằng NLP: Tách từ → Gắn nhãn từ loại → Nhận dạng thực thể có tên</p>', unsafe_allow_html=True)
    
    # Add a beautiful info box
    st.markdown("""
    <div class="success-box">
        <h4>🎯 Chào mừng đến với Ứng Dụng Phân Tích Văn Bản!</h4>
        <p>Ứng dụng này sử dụng công nghệ NLP tiên tiến để phân tích văn bản tiếng Anh. 
        Bạn có thể tách từ, gắn nhãn từ loại và nhận dạng các thực thể có tên một cách dễ dàng.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load spaCy model
    nlp = load_spacy_model()
    if nlp is None:
        st.stop()
    
    # Sidebar for input
    # st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    
    # Enhanced header with larger text and styling
    st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; padding: 1.5rem; 
                background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); 
                border-radius: 15px; box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                border: 2px solid #667eea;">
        <h1 style="color: #667eea; font-size: 2rem; font-weight: 700; margin: 0; 
                   text-shadow: 0 2px 4px rgba(102, 126, 234, 0.2);">
            📖 Nhập Văn Bản
        </h1>
        <p style="color: #64748b; font-size: 1rem; margin: 0.5rem 0 0 0; 
                  font-weight: 500;">
            Bắt đầu phân tích văn bản của bạn
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample text option
    sample_texts = {
        "Chọn văn bản mẫu": "",
        "Bài báo tin tức": "Apple Inc. announced yesterday that Tim Cook, the CEO, will visit New York City on March 15th, 2024. The company plans to invest $2 billion in artificial intelligence research.",
        "Văn bản khoa học": "Dr. Sarah Johnson from MIT published a groundbreaking study in Nature journal. The research was conducted in Boston, Massachusetts and involved 500 participants from various organizations.",
        "Báo cáo kinh doanh": "Microsoft Corporation reported quarterly earnings of $52.7 billion, representing a 12% increase year-over-year. The company's stock price rose to $350 per share on the NASDAQ exchange."
    }
    
    selected_sample = st.sidebar.selectbox("Chọn văn bản mẫu:", list(sample_texts.keys()))
    
    # Text input
    if selected_sample != "Chọn văn bản mẫu":
        default_text = sample_texts[selected_sample]
    else:
        default_text = ""
    
    text_input = st.sidebar.text_area(
        "Nhập văn bản của bạn:",
        value=default_text,
        height=200,
        help="Nhập một đoạn văn bản để phân tích"
    )
    
    # Process button
    if st.sidebar.button("🔍 Phân Tích Văn Bản", type="primary"):
        if not text_input.strip():
            st.error("Vui lòng nhập văn bản để phân tích!")
        else:
            # Show loading animation
            with st.spinner("🔄 Đang phân tích văn bản..."):
                # Process the text
                doc = nlp(text_input)
            
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4 = st.tabs(["🔤 Tách Từ", "🏷️ Gắn Nhãn Từ Loại", "👥 Nhận Dạng Thực Thể", "📊 Tổng Kết"])
            
            with tab1:
                st.header("🔤 Kết Quả Tách Từ")
                st.markdown("""
                <div class="highlight-box">
                    <h4>📝 Thông tin về Tokenization</h4>
                    <p>Tokenization là quá trình chia văn bản thành các đơn vị nhỏ nhất (tokens) như từ, dấu câu, số, v.v. 
                    Đây là bước đầu tiên trong pipeline NLP.</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("**Các từ đơn lẻ được trích xuất từ văn bản:**")
                
                # Display tokens in a nice format
                tokens_data = []
                for i, token in enumerate(doc):
                    tokens_data.append({
                        "STT": i + 1,
                        "Từ": token.text,
                        "Khoảng trắng": "Có" if token.whitespace_ else "Không",
                        "Chữ cái": "Có" if token.is_alpha else "Không",
                        "Số": "Có" if token.is_digit else "Không",
                        "Dấu câu": "Có" if token.is_punct else "Không"
                    })
                
                df_tokens = pd.DataFrame(tokens_data)
                st.dataframe(df_tokens, use_container_width=True)
                
                # Token statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Tổng số từ", len(doc))
                with col2:
                    st.metric("Từ chữ cái", len([t for t in doc if t.is_alpha]))
                with col3:
                    st.metric("Số", len([t for t in doc if t.is_digit]))
                with col4:
                    st.metric("Dấu câu", len([t for t in doc if t.is_punct]))
            
            with tab2:
                st.header("🏷️ Gắn Nhãn Từ Loại")
                st.markdown("""
                <div class="highlight-box">
                    <h4>🏷️ Thông tin về POS Tagging</h4>
                    <p>Part-of-Speech (POS) Tagging là quá trình gắn nhãn từ loại cho mỗi từ trong câu như danh từ, động từ, tính từ, v.v. 
                    Điều này giúp hiểu cấu trúc ngữ pháp của câu.</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("**Nhãn từ loại cho mỗi từ:**")
                
                # Display POS tags
                pos_data = []
                for token in doc:
                    pos_data.append({
                        "Từ": token.text,
                        "Nhãn từ loại": token.pos_,
                        "Mô tả": spacy.explain(token.pos_),
                        "Nhãn chi tiết": token.tag_,
                        "Mô tả chi tiết": spacy.explain(token.tag_),
                        "Từ gốc": token.lemma_,
                        "Từ dừng": "Có" if token.is_stop else "Không"
                    })
                
                df_pos = pd.DataFrame(pos_data)
                st.dataframe(df_pos, use_container_width=True)
                
                # POS tag distribution
                pos_counts = Counter([token.pos_ for token in doc])
                pos_df = pd.DataFrame(list(pos_counts.items()), columns=['Nhãn từ loại', 'Số lượng'])
                pos_df['Mô tả'] = pos_df['Nhãn từ loại'].apply(lambda x: spacy.explain(x))
                
                fig = px.bar(pos_df, x='Nhãn từ loại', y='Số lượng', 
                           title="Phân bố các nhãn từ loại",
                           hover_data=['Mô tả'])
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.header("👥 Nhận Dạng Thực Thể Có Tên")
                st.markdown("""
                <div class="highlight-box">
                    <h4>👥 Thông tin về Named Entity Recognition (NER)</h4>
                    <p>Named Entity Recognition là quá trình nhận dạng và phân loại các thực thể có tên trong văn bản như tên người, 
                    tổ chức, địa điểm, ngày tháng, v.v. Các thực thể được tô màu để dễ nhận biết.</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("**Các thực thể có tên được tìm thấy trong văn bản:**")
                
                # Extract entities
                entities = []
                for ent in doc.ents:
                    entities.append({
                        "Văn bản": ent.text,
                        "Loại": ent.label_,
                        "Mô tả": spacy.explain(ent.label_),
                        "Vị trí bắt đầu": ent.start_char,
                        "Vị trí kết thúc": ent.end_char,
                        "Độ tin cậy": "Cao"  # spaCy doesn't provide confidence scores by default
                    })
                
                # Create entities list for highlighting (with lowercase keys)
                entities_for_highlighting = []
                for ent in doc.ents:
                    entities_for_highlighting.append({
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char
                    })
                
                if entities:
                    df_entities = pd.DataFrame(entities)
                    st.dataframe(df_entities, use_container_width=True)
                    
                    # Entity type distribution
                    entity_counts = Counter([ent.label_ for ent in doc.ents])
                    entity_df = pd.DataFrame(list(entity_counts.items()), columns=['Loại thực thể', 'Số lượng'])
                    entity_df['Mô tả'] = entity_df['Loại thực thể'].apply(lambda x: spacy.explain(x))
                    
                    fig = px.pie(entity_df, values='Số lượng', names='Loại thực thể',
                               title="Phân bố các loại thực thể",
                               hover_data=['Mô tả'])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Highlighted text
                    st.subheader("Văn bản với thực thể được tô màu")
                    highlighted = highlight_entities(text_input, entities_for_highlighting)
                    st.markdown(highlighted, unsafe_allow_html=True)
                else:
                    st.info("Không tìm thấy thực thể có tên nào trong văn bản.")
            
            with tab4:
                st.header("📊 Tổng Kết Phân Tích")
                st.markdown("""
                <div class="success-box">
                    <h4>📊 Thống kê tổng quan</h4>
                    <p>Dưới đây là các thống kê tổng quan về văn bản đã phân tích, bao gồm số lượng từ, câu, 
                    thực thể và các từ xuất hiện nhiều nhất.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Create summary metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📈 Thống Kê Văn Bản")
                    st.metric("Tổng ký tự", len(text_input))
                    st.metric("Tổng từ", len([t for t in doc if t.is_alpha]))
                    st.metric("Tổng câu", len(list(doc.sents)))
                    st.metric("Độ dài từ trung bình", round(sum(len(t.text) for t in doc if t.is_alpha) / len([t for t in doc if t.is_alpha]), 2) if len([t for t in doc if t.is_alpha]) > 0 else 0)
                
                with col2:
                    st.subheader("🏷️ Tổng Kết Thực Thể")
                    st.metric("Tổng thực thể", len(doc.ents))
                    st.metric("Loại thực thể duy nhất", len(set(ent.label_ for ent in doc.ents)))
                    st.metric("Thực thể phổ biến nhất", max([ent.label_ for ent in doc.ents], key=[ent.label_ for ent in doc.ents].count) if doc.ents else "Không có")
                    st.metric("Thực thể dài nhất", max([ent.text for ent in doc.ents], key=len) if doc.ents else "Không có")
                
                # Word frequency
                st.subheader("📊 Từ Xuất Hiện Nhiều Nhất")
                word_freq = Counter([token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop])
                if word_freq:
                    freq_df = pd.DataFrame(list(word_freq.most_common(10)), columns=['Từ', 'Tần suất'])
                    fig = px.bar(freq_df, x='Từ', y='Tần suất', title="Top 10 từ xuất hiện nhiều nhất")
                    st.plotly_chart(fig, use_container_width=True)
    
    # Instructions
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Hướng Dẫn Sử Dụng")
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 100%); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
        <h4 style="color: #0369a1; margin-top: 0;">🚀 Cách sử dụng:</h4>
        <ol style="color: #0f172a; line-height: 1.6;">
            <li><strong>Nhập văn bản</strong> vào ô nhập liệu</li>
            <li><strong>Nhấn nút</strong> "Phân Tích Văn Bản"</li>
            <li><strong>Xem kết quả</strong> trong các tab:
                <ul style="margin-top: 0.5rem;">
                    <li>🔤 <strong>Tách Từ</strong>: Các từ đơn lẻ</li>
                    <li>🏷️ <strong>Gắn Nhãn Từ Loại</strong>: Phân loại ngữ pháp</li>
                    <li>👥 <strong>Nhận Dạng Thực Thể</strong>: Người, địa điểm, tổ chức</li>
                    <li>📊 <strong>Tổng Kết</strong>: Thống kê tổng quan</li>
                </ul>
            </li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Tips and Tricks
    st.sidebar.markdown("### 💡 Mẹo Sử Dụng")
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #fef3c7 0%, #fef7cd 100%); 
                padding: 1rem; border-radius: 10px; margin: 1rem 0; border-left: 4px solid #f59e0b;">
        <h4 style="color: #92400e; margin-top: 0; font-size: 1rem;">✨ Mẹo hay:</h4>
        <p style="color: #451a03; margin: 0.3rem 0; font-size: 0.85rem;">
            • Sử dụng văn bản mẫu để thử nghiệm
        </p>
        <p style="color: #451a03; margin: 0.3rem 0; font-size: 0.85rem;">
            • Văn bản dài cho kết quả chi tiết hơn
        </p>
        <p style="color: #451a03; margin: 0.3rem 0; font-size: 0.85rem;">
            • Hover vào thực thể để xem thông tin
        </p>
        <p style="color: #451a03; margin: 0.3rem 0; font-size: 0.85rem;">
            • Xem biểu đồ để hiểu phân bố
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Technology Info
    st.sidebar.markdown("### 🛠️ Công Nghệ")
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #faf5ff 100%); 
                padding: 1rem; border-radius: 10px; margin: 1rem 0; border-left: 4px solid #8b5cf6;">
        <h4 style="color: #6b21a8; margin-top: 0; font-size: 1rem;">⚡ Powered by:</h4>
        <p style="color: #581c87; margin: 0.3rem 0; font-size: 0.85rem;">
            <strong>spaCy</strong> - NLP Engine
        </p>
        <p style="color: #581c87; margin: 0.3rem 0; font-size: 0.85rem;">
            <strong>Streamlit</strong> - Web Framework
        </p>
        <p style="color: #581c87; margin: 0.3rem 0; font-size: 0.85rem;">
            <strong>Plotly</strong> - Visualizations
        </p>
        <p style="color: #581c87; margin: 0.3rem 0; font-size: 0.85rem;">
            <strong>Pandas</strong> - Data Processing
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <h3 style="color: #667eea; margin-bottom: 1rem;">🎉 Cảm ơn bạn đã sử dụng!</h3>
        <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
            <strong>Được xây dựng với ❤️ bằng spaCy và Streamlit</strong>
        </p>
        <p style="color: #64748b; font-size: 0.9rem;">
            Ứng dụng phân tích văn bản tiên tiến với công nghệ NLP hiện đại
        </p>
        <div style="margin-top: 1rem;">
            <span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 0.3rem 0.8rem; border-radius: 15px; 
                        font-size: 0.8rem; margin: 0 0.2rem;">
                🔤 Tokenization
            </span>
            <span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 0.3rem 0.8rem; border-radius: 15px; 
                        font-size: 0.8rem; margin: 0 0.2rem;">
                🏷️ POS Tagging
            </span>
            <span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 0.3rem 0.8rem; border-radius: 15px; 
                        font-size: 0.8rem; margin: 0 0.2rem;">
                👥 NER
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()