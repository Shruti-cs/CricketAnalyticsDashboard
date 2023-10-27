import streamlit as st
# Set the title of the Streamlit app
st.set_page_config(
    page_title="Cricket Analytics Dashboard",
    page_icon="images\hitter.png",  # You can set a favicon here if needed
    layout="wide" ,   # You can customize the layout as well
)


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


st.write('App by : Shruti C S ; Deepak Raj A')




