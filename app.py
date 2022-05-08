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
    menu_items={
        'Get Help': 'https://github.com/sonnyrd',
        'Report a bug': "https://github.com/sonnyrd",
        'About': "# V Search"
    }
)

if 'pred' not in st.session_state:
	st.session_state.pred = 0


# def change image
def change_image():
    st.session_state.pred = 0
    

col4, col5, col6 = st.columns([1.5, 1.5, 1.5])
with col5:
    st.image('Logo.jpg', use_column_width=True)
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

# Image Upload Option
choose = st.selectbox("Choose an option", ["Upload Image", "From Camera"],on_change=change_image)

if choose == "Upload Image":  
    file = st.file_uploader("Choose an image...", type=["jpg","jpeg"])
    if file is not None:
        dummy = file.name
        img = Image.open(file)
   
else:  
   picture = st.camera_input("Take a picture")

# check if image not none
if img is not None:
    col1, col2, col3  = st.columns([1,0.15,1])
    with col2:
        if st.button("Detect!"):
            st.session_state.pred = 1


col4, col5, col6 = st.columns([1.5, 1.5, 1.5])
if st.session_state.pred == 0 and img is not None:
    # with col4:
    #     st.image('Logo.jpg', use_column_width=True)
    with col5:
        st.image(img, use_column_width=True)

# Prediction Section
if st.session_state.pred == 1 and img is not None:
    col4, col5, col6 = st.columns([1.5, 1.5, 1.5])
    with col4:
        results = model(img, size=640)
        img_pred = Image.fromarray(results.render()[0])
        st.image(img_pred,use_column_width=True)
    with col5:
        list_detected = results.pandas().xyxy[0]['name'].to_list()
        select_class = st.selectbox("Choose an option", list_detected) 
        if st.session_state.pred == 1 and select_class is None :
             st.write("Sorry! we cant detect the object, please change the image and try again.")
        
        crop_detected = results.pandas().xyxy[0]
        if select_class is not None:
            xmin, ymin, xmax, ymax = crop_detected.iloc[list_detected.index(select_class)][:4]
            img_crop = np.asarray(img)[round(ymin):round(ymax),round(xmin):round(xmax)]
            img_crop = Image.fromarray(img_crop)
            st.image(img_crop)




# similarity recommendation ##########################################
if img_crop is not None:
    inputDim = (640,640)
    transformationForCNNInput = transforms.Compose([transforms.Resize(inputDim)])
    img_transform = transformationForCNNInput(img_crop)


# load the vektor pickle
if select_class is not None:
    with open(f"vector_pickle/allVectors_{select_class}.pkl", "rb") as f:
        allVectors_test = pickle.load(f)

# define Img2VecResnet18
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
        

# generate vectors for all the images in the set
img2vec = Img2VecResnet18() 


# get data inference vector
if img_transform is not None:
    allVectors_test['data_inf.jpeg'] = img2vec.getVec(img_transform)


# get similarity matrix
def getSimilarityMatrix(vectors):
    v = np.array(list(vectors.values())).T
    sim = np.inner(v.T, v.T) / ((np.linalg.norm(v, axis=0).reshape(-1,1)) * ((np.linalg.norm(v, axis=0).reshape(-1,1)).T))
    keys = list(vectors.keys())
    matrix = pd.DataFrame(sim, columns = keys, index = keys)
    
    return matrix

if allVectors_test is not None:
    similarityMatrix = getSimilarityMatrix(allVectors_test)


if similarityMatrix is not None:
# get 5 most similar images
    k = 11 # the number of top similar images to be stored

    similarNames = pd.DataFrame(index = similarityMatrix.index, columns = range(k))
    similarValues = pd.DataFrame(index = similarityMatrix.index, columns = range(k))

    for j in range(similarityMatrix.shape[0]):
        kSimilar = similarityMatrix.iloc[j, :].sort_values(ascending = False).head(k)
        similarNames.iloc[j, :] = list(kSimilar.index)
        similarValues.iloc[j, :] = kSimilar.values

if similarNames is not None:
    st.write(" ")

    st.markdown("""
    <style>
    b{  background: #E0DDEC;
   padding: 5px 5px 5px 5px;
   border-radius: 2px;
   box-shadow: 0 0 5px 0 rgba(0, 0, 0, 0.2), 0 5px 5px 0 rgba(0, 0, 0, 0.24);
   color: #888
   margin: 0 0 0 0;
   font-size: 50px;
   line-height: 0.80;
   width: 200px;
   text-align: center;} </style>

    <b> 
    Our similar catalogue 
    </b> """, unsafe_allow_html=True)

    # st.markdown("""<br>""")
    st.markdown(""" """)
    st.markdown(""" """)

    col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
    with col1:
        st.image(Image.open(f"catalog/{select_class}/{similarNames.loc['data_inf.jpeg'][1]}"), use_column_width=True)
    #     with st.expander("See explanation"):
    #         st.write("""
    #      The chart above shows some numbers I picked for you.
    #      I rolled actual dice for these, so they're *guaranteed* to
    #      be random.
    #  """)
    with col2:
        st.image(Image.open(f"catalog/{select_class}/{similarNames.loc['data_inf.jpeg'][2]}"),use_column_width=True)
    with col3:
        st.image(Image.open(f"catalog/{select_class}/{similarNames.loc['data_inf.jpeg'][3]}"),use_column_width=True)
    with col4:
        st.image(Image.open(f"catalog/{select_class}/{similarNames.loc['data_inf.jpeg'][4]}"),use_column_width=True)
    with col5:
        st.image(Image.open(f"catalog/{select_class}/{similarNames.loc['data_inf.jpeg'][5]}"),use_column_width=True)

    
    a, b, c = st.columns([1.5, 1.5, 1.5])
    with b:
        next_ = st.button('Show more')

    if next_ :
            col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
            with col1:
                st.image(Image.open(f"catalog/{select_class}/{similarNames.loc['data_inf.jpeg'][6]}"), use_column_width=True)
            with col2:
                st.image(Image.open(f"catalog/{select_class}/{similarNames.loc['data_inf.jpeg'][7]}"),use_column_width=True)
            with col3:
                st.image(Image.open(f"catalog/{select_class}/{similarNames.loc['data_inf.jpeg'][8]}"),use_column_width=True)
            with col4:
                st.image(Image.open(f"catalog/{select_class}/{similarNames.loc['data_inf.jpeg'][9]}"),use_column_width=True)
            with col5:
                st.image(Image.open(f"catalog/{select_class}/{similarNames.loc['data_inf.jpeg'][10]}"),use_column_width=True)



