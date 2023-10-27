import streamlit as st
import base64
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set the title of the Streamlit app
st.set_page_config(
    page_title="Cricket Analytics Dashboard",
    page_icon="images/hitter.png",  # You can set a favicon here if needed
    layout="wide" ,   # You can customize the layout as well
)





@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("images/Sbg2.jpg")
img1 = get_img_as_base64("images/side.jpeg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{img}");
background-size: 100% 100%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img1}");
background-position: center; 

background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

with st.container():
    st.title("CRICKET ANALYTICS DASHBOARD")
    # Define a Streamlit app
    st.header('Switch between Cricket Match Analysis')
    # Create a button with custom size
    button_label1 = "WorldCup"
    button_style1 = "padding: 10px 20px; font-size: 16px;"
    b1 = st.button(label=button_label1, key="custom_button1", help="Custom-sized Button", on_click=None, args=None, kwargs=None, disabled=False)

    # Create a button that will load the first Power BI report
    if b1:
        st.markdown(
            f'<iframe title="T20_WorldCup_Analysis" width="100%" height="541.25" src="https://app.powerbi.com/reportEmbed?reportId=c7775c23-a9c5-4417-893b-08c3c68a7168&autoAuth=true&ctid=8c25f501-4f7e-4b41-9aa6-bdd5f317d4e4" frameborder="0" allowFullScreen="true"></iframe> <style>div[data-testid="stButton.custom_button1"] button {{ {button_style1} }}</style>',
            unsafe_allow_html=True
        )

    # Create a button with custom size
    button_label = "IPL"
    button_style = "padding: 10px 20px; font-size: 16px;"
    b2 = st.button(label=button_label, key="custom_button", help="Custom-sized Button", on_click=None, args=None, kwargs=None, disabled=False)

    # Create a button that will load the second Power BI report
    if b2:
        st.markdown(
            f'<iframe title="ipl" width="100%" height="541.25" src="https://app.powerbi.com/reportEmbed?reportId=af7ba4e8-e975-4cc6-9c7f-017b56341def&autoAuth=true&ctid=8c25f501-4f7e-4b41-9aa6-bdd5f317d4e4" frameborder="0" allowFullScreen="true"></iframe> <style>div[data-testid="stButton.custom_button"] button {{ {button_style} }}</style>',
            unsafe_allow_html=True
        )






# Title
st.sidebar.header('Cricket Analytics')

# File Uploader
uploaded_file = st.sidebar.file_uploader("Upload a dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Display first few rows of the dataset
    st.subheader('Dataset Overview')
    st.dataframe(data.head())

    # Data Cleaning
    st.subheader('Data Cleaning')
    
    # Handle missing values
    missing_values = data.isnull().sum()
    st.write('### Missing Values')
    st.write(missing_values)
    
    # Drop duplicate rows
    data.drop_duplicates(inplace=True)
    st.write('### Duplicate Rows Removed')
    
    # Summary statistics after cleaning
    st.subheader('Summary Statistics (After Cleaning)')
    st.write(data.describe())

    # Data Visualization
    st.subheader('Data Visualization')

    # Select a column for visualization
    feature = st.selectbox('Select a feature:', data.columns)

    # Histogram
    st.write('### Histogram')
    plt.figure(figsize=(8, 6))
    sns.histplot(data[feature], bins=20, kde=True)
    st.pyplot(plt)

    # Pair plot (scatter matrix)
    st.write('### Pair Plot')
    sns.pairplot(data)
    st.pyplot(plt)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    correlation = data.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    st.pyplot()

    # Distribution of features
    st.subheader("Distribution of Features")
    feature_cols = data.columns[:-1]
    for col in feature_cols:
        st.write(f"### {col} Distribution")
        sns.histplot(data[col], kde=True)
        st.pyplot()

    # Box plots
    st.subheader("Box Plots")
    for col in feature_cols:
        st.write(f"### {col} by Species")
        sns.boxplot(x='species', y=col, data=data)
        st.pyplot()

    # Violin plots
    st.subheader("Violin Plots")
    for col in feature_cols:
        st.write(f"### {col} by Species")
        sns.violinplot(x='species', y=col, data=data)
        st.pyplot()

    # Correlation with a target variable
    st.subheader("Correlation with Target")
    correlation_with_target = data.corr()['sepal_length'].sort_values(ascending=False)
    st.bar_chart(correlation_with_target)

    # Correlation scatter plots
    st.subheader("Correlation Scatter Plots")
    for col in feature_cols:
        st.write(f"### Sepal Length vs. {col}")
        sns.scatterplot(x='sepal_length', y=col, data=data)
        st.pyplot()


    # Model Building
    st.subheader('Model Building')

    # Select features and target variable
    X = data[[feature]]
    y = data['target_column']  # Replace 'target_column' with your target variable

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Display model results
    st.write('### Model Results')
    st.write(f'Mean Squared Error: {mse:.2f}')
    st.write(f'R-squared: {r2:.2f}')


st.write('App by : Shruti C S ; Deepak Raj A')




