import streamlit as st 
from PIL import Image
from classify import predict

def converter(value):
    if value == 0:
        return "NORMAL"
    else:
        return "PNEUMONIA"

st.title("Upload + X-RAY IMAGE")
html_temp="""
<div style="background-color:gray;padding:15px;">
<h2>X-RAY IMAGE OF THE LUNG</h2>
</div>
"""
st.markdown(html_temp,unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=("jpeg","jpg","png"))
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    converter = predict(uploaded_file)
    st.write('I THINK:', converter)




