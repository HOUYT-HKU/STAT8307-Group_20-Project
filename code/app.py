# app.py
###

###终端运行
###pip install streamlit matplotlib networkx
###streamlit run app.py

###
import streamlit as st
from KeywordExtractor import KeywordExtractor
from nlp_utils import prune_text
import matplotlib.pyplot as plt
import networkx as nx

# 页面标题和说明
st.title("Graph-Based Keyword Extractor")
st.markdown("""
- Enter text → select algorithm → extract keywords → visualise graphs
- Supporting algorithms: degree_centrality,closeness_centrality,betweenness_centrality,
     eigenvector_centrality,pagerank
""")

# 侧边栏配置参数
st.sidebar.header("parameter setting")
window_size = st.sidebar.slider("Sliding window size", 2, 10, 3)
method = st.sidebar.selectbox(
    "Selection sorting algorithm",
    ("degree_centrality", "closeness_centrality", "betweenness_centrality",
     "eigenvector_centrality", "pagerank")
)
num_keywords = st.sidebar.slider("Display Keyword Quantity", 1, 20, 5)

# 主界面输入框
abstract = st.text_area("Enter text", height=200, value="""The island country of Japan has developed into a great economy after World War 2. 
The Japan sea is a source of fish. Sushi is a famous fish and rice food.""")

# 处理按钮
if st.button("Keywords Extraction"):
    if not abstract.strip():
        st.error("Enter text！")
    else:
        try:
            # 初始化关键词提取器
            ke = KeywordExtractor(abstract=abstract, window_size=window_size)

            # 显示预处理后的 Tokens
            st.subheader("Pre-processing results")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Tokens**", ke.tokens)
            with col2:
                st.write("**Sentences**", ke.sentences)

            # 提取关键词
            keyword_dict = ke.order_nodes(method=method, to_print=False)
            keywords = list(keyword_dict.keys())[:num_keywords]

            # 显示关键词
            st.subheader("Extraction results")
            st.write(f"**Top {num_keywords} Keywords(algorithm：{method})**")
            st.write(keywords)

            # 可视化图形
            st.subheader("Visualization results")
            plt.figure(figsize=(10, 8))
            pos = nx.circular_layout(ke.graph)
            nx.draw(ke.graph, pos, with_labels=True, node_size=800,
                    node_color='lightblue', font_size=10, font_weight='bold')
            edge_labels = nx.get_edge_attributes(ke.graph, 'weight')
            nx.draw_networkx_edge_labels(ke.graph, pos, edge_labels=edge_labels)
            st.pyplot(plt)

        except Exception as e:
            st.error(f"err：{str(e)}")