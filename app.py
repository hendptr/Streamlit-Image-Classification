import torch
import streamlit as st
from models import Network
from utils import predict
from PIL import Image
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title('Image Classification')
st.write("""
Choose your model first
 """)
st.write('Backend Use : ', device)

model_user = st.selectbox('Select Model to Use', ('Resnet18', 'Resnet34', 'Resnet101', 'VGG16', 'VGG19', 'GoogleNet'))
img_user = st.file_uploader("Upload an image", type="jpg")
st.write('Model : ', model_user)
model = Network(model_user)

num_prob = st.sidebar.slider('Number Of Probability', 1, 10)
st.write('Number Probability : ',num_prob)
if img_user:
    image = Image.open(img_user)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    preds = predict(model, image, num_prob, device)
    df = pd.DataFrame(preds, columns=['Prediction', 'Probability (%)'])
    st.table(df)



