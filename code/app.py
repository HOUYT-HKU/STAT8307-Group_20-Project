# app.py
###

###pip install streamlit matplotlib networkx
###streamlit run app.py

###
import streamlit as st
from KeywordExtractor import KeywordExtractor
from nlp_utils import prune_text
import matplotlib.pyplot as plt
import networkx as nx


st.title("Graph-Based Keyword Extractor")
st.markdown("""
- Enter text → select algorithm → extract keywords → visualise graphs
- Supporting algorithms: degree_centrality,closeness_centrality,betweenness_centrality,
     eigenvector_centrality,pagerank
""")


st.sidebar.header("parameter setting")
window_size = st.sidebar.slider("Sliding window size", 2, 10, 3)
method = st.sidebar.selectbox(
    "Selection sorting algorithm",
    ("degree_centrality", "closeness_centrality", "betweenness_centrality",
     "eigenvector_centrality", "pagerank")
)
num_keywords = st.sidebar.slider("Display Keyword Quantity", 1, 20, 5)


abstract = st.text_area("Enter text", height=200, value="""The island country of Japan has developed into a great economy after World War 2. 
The Japan sea is a source of fish. Sushi is a famous fish and rice food.""")


if st.button("Keywords Extraction"):
    if not abstract.strip():
        st.error("Enter text！")
    else:
        try:
            
            ke = KeywordExtractor(abstract=abstract, window_size=window_size)

          
            st.subheader("Pre-processing results")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Tokens**", ke.tokens)
            with col2:
                st.write("**Sentences**", ke.sentences)

         
            keyword_dict = ke.order_nodes(method=method, to_print=False)
            keywords = list(keyword_dict.keys())[:num_keywords]

          
            st.subheader("Extraction results")
            st.write(f"**Top {num_keywords} Keywords(algorithm：{method})**")
            st.write(keywords)

         
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
