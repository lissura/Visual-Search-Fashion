import streamlit as st
import numpy as np
from PIL import Image
import torch
import requests
from io import BytesIO

# similarity
from torchvision import transforms, models
import pickle
import pandas as pd



st.set_page_config(
    page_title="V Search",
    page_icon="💃",
    layout="wide",
    initial_sidebar_state="expanded",

)

# def change image
def change_image():
    st.session_state.pred = 0
    st.session_state.page = 0
    img = None

def increment_counter():
	st.session_state.page += 5

def decrement_counter():
	st.session_state.page -= 5

if 'pred' not in st.session_state:
	st.session_state.pred = 0

if 'page' not in st.session_state:
    st.session_state.page = 0


# Load the model
model = torch.hub.load(
'yolov5',
'custom', 
 path='saved_model/best.pt',
 source='local')  # local repo

model.conf = 0.25
model.multi_label = True

# Initialize variable
img = None
img_crop = None
select_class = None
img_transform = None
allVectors_test = None
similarityMatrix = None
similarNames = None
select_catalog = None

 

# define column
col1, col2, col3, col4, col5  = st.columns([1,0.5,1,0.5,1])


# logo
with col3:
    st.image('Logo2.jpg', use_column_width=True)
st.markdown(
    """
    <style>
    h2 {background-color:#000000;
        font-size:80px;
        color: #FFFFFF;
        border: 2px solid;
        padding: 20px 20px 20px 70px; */
        padding: 5% 5% 5% 10%;
    text-align: center
    }
    </style>
    <h2> Your virtual search helper </h2>
    
    ----
    """, unsafe_allow_html=True)
st.info("Only available for : Blouse, Cardigan, Dress, Hoodie,\
         Trousers, Skirts, and Jeans catalogue")
st.info("This website is still under development, now we're only able to\
     search womens catalogue")
#define column
col6, col7, col8, col9, col10 = st.columns([1,0.5,1.5,0.5,1])

# Image Upload Option

with col6:
    choose = st.selectbox("Choose an option", ["Upload Image","From URL","From Camera"],on_change=change_image)
   
    if choose == "Upload Image":  
        file = st.file_uploader("Choose an image...", type=["jpg","jpeg"])
        
        if file is not None:
            dummy = file.name
            img = Image.open(file)

    if choose == "From URL":
        url = st.text_area("Enter URL", placeholder="Paste the image URL here...")
        if url:
            try: 
                response = requests.get(url)
                dummy = url
                img = Image.open(BytesIO(response.content))
            except:  
                st.error(
                    "Invalid URL!!!! Please use a different URL or upload an image."
                )

    if choose == 'From Camera':
        file = st.camera_input("Take a picture")

        if file is not None:
            # To read image file buffer as a PIL Image:
            dummy = file.name
            img = Image.open(file)
    
    if img is not None: 
        if st.button("Detect!"):
            st.session_state.pred = 1


if st.session_state.pred == 0 and img is not None:
    with col8:
        st.image(img, width=640)


# Prediction Section
if st.session_state.pred == 1 and img is not None:
    with col8:
        results = model(img, size=640)
        img_pred = Image.fromarray(results.render()[0])
        st.image(img_pred, width=640)
    
    
    with col10:
        list_detected = results.pandas().xyxy[0]['name'].to_list()
        select_class = st.selectbox("Choose an option", list_detected)
        if st.session_state.pred == 1 and select_class is None :
             st.error("Sorry! we cant detect the object, please change the image and try again.")

       
        if select_class is not None:
            crop_detected = results.pandas().xyxy[0]
            xmin, ymin, xmax, ymax = crop_detected.iloc[list_detected.index(select_class)][:4]
            img_crop = np.asarray(img)[round(ymin):round(ymax),round(xmin):round(xmax)]
            img_crop = Image.fromarray(img_crop)
            st.image(img_crop, width=360)





# similarity recommendation ##########################################
if select_class is not None:

    st.write(" ")

    st.markdown("""
    <style>
    box{  background: #E0DDEC;
   padding: 5px 5px 5px 5px;
   border-radius: 2px;
   box-shadow: 0 0 5px 0 rgba(0, 0, 0, 0.2), 0 5px 5px 0 rgba(0, 0, 0, 0.24);
   color: #888
   margin: center;
   font-size: 50px;
   line-height: 0.80;
   width: 200px;
   text-align: center;} 
   </style>
   
    <box> 
    Our similar catalogue 
    </box> """, unsafe_allow_html=True)

    # st.markdown("""<br>""")
    st.markdown(""" """)
    st.markdown(""" """)

# select item to be recommended
if select_class is not None:
    col11, col12, col13, col14, col15 = st.columns([1,0.5,3,0.4,0.3])
    with col11:
        select_catalog = st.multiselect('filter', list_detected,default=list_detected)
    
        if select_catalog != []:
            with col15:
                st.write("")
                st.write("")
                if st.session_state.page == 10:
                    next_b = st.button("next page!",disabled=True, help="You have reached the end of the list.")
                else:
                    next_b = st.button("next page!",on_click=increment_counter)

            with col14:
                st.write("")
                st.write("")
                if st.session_state.page == 0:
                    prev_b = st.button("previous page!",disabled=True, help="You have reached the beginning of the list.")
                else:
                    prev_b = st.button("previous page!",on_click=decrement_counter)

# convert to tensor
class Img2VecResnet18():
    def __init__(self):
        
        self.device = torch.device("cpu")
        self.numberFeatures = 512
        self.modelName = "resnet-18"
        self.model, self.featureLayer = self.getFeatureLayer()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.toTensor = transforms.ToTensor()
        
        # normalize the resized images as expected by resnet18
        # [0.485, 0.456, 0.406] --> normalized mean value of ImageNet, [0.229, 0.224, 0.225] std of ImageNet
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    def getVec(self, img):
        image = self.normalize(self.toTensor(img)).unsqueeze(0).to(self.device)
        embedding = torch.zeros(1, self.numberFeatures, 1, 1)

        def copyData(m, i, o): embedding.copy_(o.data)

        h = self.featureLayer.register_forward_hook(copyData)
        self.model(image)
        h.remove()

        return embedding.numpy()[0, :, 0, 0]

    def getFeatureLayer(self):
        
        cnnModel = models.resnet18(pretrained=True)
        layer = cnnModel._modules.get('avgpool')
        self.layer_output_size = 512
        
        return cnnModel, layer

# cossine similarity
class cossine_similarity():
    def get_img_crop(self):
        xmin, ymin, xmax, ymax = crop_detected.iloc[list_detected.index(selected_catalog)][:4]
        img_crop = np.asarray(img)[round(ymin):round(ymax),round(xmin):round(xmax)]
        img_crop = Image.fromarray(img_crop)
        return img_crop

    def transform_image_crop(self):
        inputDim = (640,640)
        transformationForCNNInput = transforms.Compose([transforms.Resize(inputDim)])
        img_crop = self.get_img_crop()
        img_transform = transformationForCNNInput(img_crop)
        return img_transform
    
    def get_vectors_pickle(self):
        with open(f"vector_pickle/allVectors_{selected_catalog}.pkl", "rb") as f:
            allVectors_test = pickle.load(f)
        return allVectors_test
    
    def generate_vector(self):
        img2vec = Img2VecResnet18()
        allVectors_test = self.get_vectors_pickle()
        img_transform = self.transform_image_crop()
        allVectors_test['data_inf.jpeg'] = img2vec.getVec(img_transform)
        return allVectors_test
    
    def getSimilarityMatrix(self):
        vectors = self.generate_vector()
        v = np.array(list(vectors.values())).T
        sim = np.inner(v.T, v.T) / ((np.linalg.norm(v, axis=0).reshape(-1,1)) * ((np.linalg.norm(v, axis=0).reshape(-1,1)).T))
        keys = list(vectors.keys())
        matrix = pd.DataFrame(sim, columns = keys, index = keys)
        return matrix
    
    def getSimilarityNames(self):
        k = 16 # the number of top similar images to be stored
        similarityMatrix = self.getSimilarityMatrix()
        similarNames = pd.DataFrame(index = similarityMatrix.index, columns = range(k))
        similarValues = pd.DataFrame(index = similarityMatrix.index, columns = range(k))

        for j in range(similarityMatrix.shape[0]):
            kSimilar = similarityMatrix.iloc[j, :].sort_values(ascending = False).head(k)
            similarNames.iloc[j, :] = list(kSimilar.index)
            similarValues.iloc[j, :] = kSimilar.values

        return similarNames


if select_catalog is not None:

    for i in range(len(select_catalog)):
        st.subheader(select_catalog[i].capitalize())
        exec(f'col_cat{i}_1, col_cat{i}_2, col_cat{i}_3, col_cat{i}_4, col_cat{i}_5 = st.columns([1,1,1,1,1])')
        selected_catalog = select_catalog[i]
        similarNames = cossine_similarity().getSimilarityNames()

        if similarNames is not None:
            if 1+st.session_state.page < 16:
                with vars()[f'col_cat{i}_1']:
                    st.image(Image.open(f"catalog/{select_catalog[i]}/{similarNames.loc['data_inf.jpeg'][1+st.session_state.page]}"), use_column_width=True)
            if 2+st.session_state.page < 16:
                with vars()[f'col_cat{i}_2']:
                    st.image(Image.open(f"catalog/{select_catalog[i]}/{similarNames.loc['data_inf.jpeg'][2+st.session_state.page]}"),use_column_width=True)
            if 3+st.session_state.page < 16:
                with vars()[f'col_cat{i}_3']:
                    st.image(Image.open(f"catalog/{select_catalog[i]}/{similarNames.loc['data_inf.jpeg'][3+st.session_state.page]}"),use_column_width=True)
            if 4+st.session_state.page < 16:
                with vars()[f'col_cat{i}_4']:
                    st.image(Image.open(f"catalog/{select_catalog[i]}/{similarNames.loc['data_inf.jpeg'][4+st.session_state.page]}"),use_column_width=True)
            if 5+st.session_state.page < 16:
                with vars()[f'col_cat{i}_5']:
                    st.image(Image.open(f"catalog/{select_catalog[i]}/{similarNames.loc['data_inf.jpeg'][5+st.session_state.page]}"),use_column_width=True)

st.write(st.session_state.pred)
