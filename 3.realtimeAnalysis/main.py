import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go 


page_bg_img = '''
<style>
body {
background-image: url("bg.jpg");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)
# Function to clean data
def clean_data(data):
    # Perform data cleaning operations here
    cleaned_data = data.dropna()  # Example: dropping rows with missing values
    return cleaned_data

# Function to load and clean data
@st.cache_resource
def load_and_clean_data(file):
    data = pd.read_csv(file)
    cleaned_data = clean_data(data)
    return cleaned_data

# Title
st.title('T20 Cricket Matches ')

col1, col2 = st.columns(2)

# File Uploader
uploaded_file = st.sidebar.file_uploader("Upload a dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    data = load_and_clean_data(uploaded_file)

    with col1:
        
    # Display first few rows of the cleaned dataset
        st.subheader('Cleaned Dataset Overview')
        st.dataframe(data.head())
        

        # Summary statistics
        st.subheader('Summary Statistics')
        st.write(data.describe())

        # Data Visualization
        st.subheader('Data Visualization')

        # Select a column for visualization
        feature = st.selectbox('Select a feature:', data.columns)

    # Histogram

    with col1:

        st.write('### Histogram')
        plt.figure(figsize=(8, 6))
        sns.histplot(data[feature], bins=20, kde=True)
        st.pyplot(plt)

    with col1:
    # Box plot
        st.write('### Box Plot')
        plt.figure(figsize=(8, 6))
        sns.boxplot(y=feature, data=data)
        st.pyplot(plt)

    with col2:
        # Pair plot (scatter matrix)
        st.write('### Pair Plot')
        sns.pairplot(data)
        st.pyplot(plt)
    with col2:
        # Correlation heatmap
        st.write('### Correlation Heatmap')
        corr_matrix = data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        st.pyplot(plt)

    with col2:
    # Create an interactive histogram with binning control using Plotly Express
        st.write('### Interactive Histogram')
        selected_column = st.selectbox('Select a column:', data.columns)
        fig_hist = px.histogram(data, x=selected_column, nbins=20, title='Histogram')
        st.plotly_chart(fig_hist)

    # # Create an interactive heatmap with color mapping using Seaborn
    # st.write('### Interactive Heatmap')
    # selected_columns = st.multiselect('Select columns for heatmap:', data.columns)
    # heatmap_data = data[selected_columns]
    # heatmap = sns.heatmap(heatmap_data.corr(), annot=True, cmap='coolwarm')
    # st.pyplot(heatmap.figure)

    # # Create an interactive line chart with a slider using Plotly Express
   

    # st.write('### Interactive Line Chart')
    # fig = px.line(data, x='Order Date', y='Sales', title='Sales Over Time')
    # st.plotly_chart(fig)

    # fig= px.line(
    #     data,
    #     x="Striker",
    #     y="Boundry",
    #     color="Scenario"
    # )
    # st.plotly_chart(fig,use_container_width=True)

   










