# -*- coding: utf-8 -*-

import copy
import streamlit as st
from streamlit_option_menu import option_menu
import PIL
from PIL import Image
from colorthief import ColorThief
import webcolors
import os
import io
import tempfile
from ultralytics import YOLO
from tensorflow.keras.applications.resnet_rs import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import random
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.image import imread
import textwrap
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import gridspec
import tensorflow_hub as tf_hub
from streamlit_extras.let_it_rain import rain
from tensorflow.keras.applications.vgg16 import preprocess_input as pinp

# Set the Streamlit page configuration
# Set the title, icon, and layout of the Streamlit app page
st.set_page_config(page_title="KTA by ColdShower team", page_icon="random", layout="wide")

# Define a caching decorator to avoid loading the image repeatedly and to speed up subsequent visits
@st.cache_data
def load_image():
    # Load an image from the given path
    image = Image.open("C:/streamlit_files/title.jpg")
    return image

# Display the image on the Streamlit page with full column width
st.image(load_image(), caption="", use_column_width=True)

# Create a sidebar in Streamlit 
with st.sidebar:
    # Option menu in the sidebar with specific icons for each menu item
    selected = option_menu("Main Menu", ['Home', 'Know Thy Art','Neural Style Transfer','Artwork MBTI'], 
        icons=['shop', 'palette','camera fill','puzzle'], menu_icon="cast", default_index=0)

# Define a caching decorator to avoid loading and resizing images repeatedly
@st.cache_data
def load_and_resize_images():
    images = []
    # Load and resize five images from the specified path
    for i in range(1, 6):
        img=Image.open(f"C:/streamlit_files/home_{i}.jpg")
        images.append(img)
    return images

# When the selected option from the sidebar is 'Home'
if selected == 'Home':
    # Display a header text on the Streamlit page
    st.header("Welcome to our Homepage!")
    
    # Load and display the resized images on the Streamlit page
    images = load_and_resize_images()
    for img in images:
        st.image(img, use_column_width=True)

# When the selected option from the sidebar is 'Know Thy Art'
elif selected == 'Know Thy Art':
    
    # Define a caching decorator to avoid loading the YOLO model repeatedly and to speed up subsequent visits
    @st.cache_resource
    def yolo():
        # Load the YOLO model from the given path
        model = YOLO(r"C:\streamlit_files\best_m.pt")
        return model
    
    # Load the YOLO model using the defined function
    model = yolo()
    
    # A form element in the Streamlit with a unique key "form"
    with st.form(key="form"):
        
        # Add a file uploader widget to the form, allowing users to upload 'png', 'jpg', or 'jpeg' files
        # The uploaded file is stored in the variable 'source_img'
        source_img = st.file_uploader(label='Choose an image...', type=['png','jpg', 'jpeg'])
        
        # Add a submit button to the form with the label "Analyze"
        # When clicked, any processing related to this form would be executed
        submit_button = st.form_submit_button(label="Analyze")
        
        # When the submit button was clicked
        if submit_button:
            
            # If an image has been uploaded
            if source_img:
                # Open the uploaded image using the Image module
                uploaded_image = Image.open(source_img)
                
                # If the uploaded_image is successfully loaded
                if uploaded_image:
                    # Use the previously loaded YOLO model to make predictions on the uploaded image
                    result = model.predict(uploaded_image)
                    # Extract and prepare the plot of the result, reversing the color channels from BGR to RGB
                    result_plot = result[0].plot()[:, :, ::-1]               
                    
                    # Display a spinner with the message "Running...." while the following code block is being executed
                    with st.spinner("Running...."):
                        try:
                            result_2 = result[0] # Extract the prediction result
                            box = result_2.boxes[0] # Get the bounding box of the detected object (for one object only)         
                            cords = box.xyxy[0].tolist() # Extract the coordinates of the bounding box
                            cords = [round(x) for x in cords] # Round the coordinates to get integer values
                            area = tuple(cords) # Convert the coordinates to a tuple which represents the area for cropping
                            
                            # Define a caching decorator to avoid reloading and cropping the image repeatedly
                            @st.cache_data
                            def load_and_crop_image(source_img, area):
                                lc_img = uploaded_image.crop(area) # Crop the image using the specified area
                                lc_img=copy.deepcopy(lc_img) # Deep copy the cropped image to ensure original data isn't modified
                                return lc_img
                            
                            # Load and crop the image using the defined function
                            cropped_img = load_and_crop_image(source_img, area)
                            
                            # Split the Streamlit layout into three columns
                            col1, col2,col3 = st.columns(3)
                            
                            # In the first column
                            with col1:
                                # Display the original uploaded image
                                st.image(image=uploaded_image,
                                         caption='Uploaded Image',
                                         use_column_width=True) # Adjust the image to fit the column width    
                            
                            # In the second column
                            with col2:
                                # Display the result image with detected objects
                                st.image(result_plot, 
                                         caption="Detection Image", 
                                         use_column_width=True)
                            
                            # In the third column
                            with col3:
                                # Display the cropped image containing only the detected object
                                st.image(cropped_img, 
                                         caption="Cropped Image", 
                                         use_column_width=True)
                                
                        # In case there's an exception/error in the above code block
                        except:
                            # Display only the original uploaded image with an alternative caption
                            st.image(image=uploaded_image,
                                     caption='No paintings detected in the uploaded image', 
                                     use_column_width=True)  
                            
                            # Define a caching decorator to ensure that the uncropped image is processed only once and reused on subsequent calls without having to deep copy the image again
                            @st.cache_data
                            def uncropped_img():
                                # Deep copy the uploaded image to ensure original data isn't modified
                                uc_img=copy.deepcopy(uploaded_image)
                                return uc_img
                            
                            # Fetch the uncropped image using the defined function
                            cropped_img=uncropped_img()
                        
                        # Define a caching decorator to ensure the model is loaded only once and reused on subsequent calls
                        @st.cache_resource
                        def rnrs50():
                            # Load a pre-trained model from the specified path
                            model=load_model(r"C:\streamlit_files\model_resnetrs50_lion_dense10240_noda.h5")
                            return model
                        
                        m = rnrs50() # Load the model using the defined function
                        x = img_to_array(cropped_img) # Convert the cropped image into an array format
                        x = tf.image.resize(x, [224, 224]) # Resize the image to match the model's input size
                        x = np.array([x]) # Expand the dimensions of the image to match the model's input shape
                        x = preprocess_input(x) # Preprocess the image using the appropriate preprocessing function for the model (from tensorflow.keras.applications.resnet_rs import preprocess_input)                
                        predict = m.predict(x) # Make a prediction using the loaded model                      
                        
                        # Define a dictionary to map class indices to their respective art style names in English
                        class_indices = {0: 'Abstract Expressionism', 1: 'Baroque', 2: 'Cubism', 3: 'Impressionism',4 : 'Primitivism',5:'Rococo',6:'Surrealism'}  # Replace with the correct class indices and labels.
                        
                        # Define a dictionary to map class indices to their respective art style names in Korean
                        korean_class_indices={0:'추상표현주의 (Abstract Expressionism)',
                                              1:'바로크 (Baroque)',
                                              2:'입체주의 (Cubism)',
                                              3:'인상주의 (Impressionism)',
                                              4:'원시주의 (Primitivism)',
                                              5:'로코코 (Rococo)',
                                              6:'초현실주의 (Surrealism)'}                    
                        
                        # Get the indices of the top 3 predicted classes in descending order of confidence
                        top_3_indices = np.argsort(predict[0])[-3:][::-1]
                        
                        # Map the top 3 indices to their respective class labels
                        top_3_labels = [class_indices[index] for index in top_3_indices]
                        
                        # Fetch the predicted probabilities for the top 3 predictions
                        top_3_probabilities = [predict[0][index] * 100 for index in top_3_indices]
                        
                        # Get the index of the top prediction
                        top_prediction_index = np.argsort(predict[0])[-1]
                        
                        # Add a divider in the Streamlit app for better visual separation
                        st.divider()
                        
                        # Split the Streamlit layout into two columns
                        col1,col2 = st.columns(2)
                        
                        # In the first column
                        with col1:
                            
                            # Use HTML to display a stylized header
                            st.markdown("<h2 style='text-align: center; color: black;'>Top 3 Predicted Classes</h2>", unsafe_allow_html=True)
                            
                            fig, ax = plt.subplots() # Create a matplotlib figure and axis for a pie chart
                            
                            # Plot a pie chart with the top 3 predicted probabilities
                            wedges, texts= ax.pie(
                                top_3_probabilities, 
                                labels=['', '', ''], # Empty labels and use a legend instead
                                startangle=90, 
                                pctdistance=0.8, 
                                labeldistance=0.7,
                                colors=['#161953','#B3CEE5','#FAC898']
                                )
                            
                            # Create a white circle to convert the pie chart into a doughnut chart
                            circle = plt.Circle((0, 0), 0.6, color='white')
                            
                            # Add the white circle to the axis
                            ax.add_artist(circle)
                            
                            # Prepare a list of labels for the pie chart, displaying the class names and their probabilities
                            top_3_info=[]
                            
                            for index in top_3_indices:
                                class_label = class_indices[index]
                                probability = predict[0][index] * 100
                                top_3_info.append(f'{class_label} ({probability:.2f}%)')
                                
                            # Add a legend to the pie chart
                            ax.legend(wedges, top_3_info, loc='lower center', fontsize=12, bbox_to_anchor=(0, -0.2, 1, 1))
                            
                            # Display the pie chart in the Streamlit
                            st.pyplot(fig)  
                        
                        # In the second column
                        with col2:
                            # Add vertical spacing using empty titles
                            st.title('')
                            st.title('')
                            
                            # Use HTML to display a stylized header
                            st.markdown("<h3 style='text-align: center; color: black;'>❣️ 해당 그림의 사조는<br></h3>", unsafe_allow_html=True)
                            
                            # Display the top predicted art style in Korean using the korean_class_indices dictionary
                            st.markdown(f"<h2 style='text-align: center; color: #161953;'>{korean_class_indices[top_prediction_index]}<br></h2>", unsafe_allow_html=True)
                            
                            # Continue the stylized message
                            st.markdown("<h3 style='text-align: center; color: black;'>와 가장 비슷합니다.<br></h3>", unsafe_allow_html=True)
                            
                            # Add more vertical spacing using an empty title
                            st.title('')
                            
                            # Define a caching decorator to ensure that the data is loaded only once and reused on subsequent calls
                            @st.cache_data
                            def styles_v4():
                                # Load a CSV file into a Pandas DataFrame
                                styles_df = pd.read_csv("C:/streamlit_files/styles_v8.csv")
                                return styles_df
                            
                            # Load the data using the defined function
                            df = styles_v4()
                            
                            # Filter rows from the DataFrame where the style matches the top predicted style
                            matching_rows = df[df['style'] == class_indices[top_prediction_index]]
                            
                            # Extract the descriptions related to the matched style
                            matching_description = matching_rows['app'].values
                            
                            # Create a 1:6 ratio split columns layout
                            col1, col2 = st.columns([1, 6])
                            
                            # Check if there are any matching descriptions
                            if len(matching_description) > 0:
                                # Display each matching descriptions in the larger column (col2)
                                for app in matching_description:
                                    col2.markdown(app,unsafe_allow_html=True)
                            else:
                                # Display a subheader if no descriptions are found for the predicted art style
                                st.subheader("No related descriptions found for the predicted art style.")
                                
                        # Try to execute the following block of code
                        try:
                            
                            # Check if the bytes representation of cropped_img is not equal to a new 5x5 white image
                            # This step ensures that the cropped_img is not an empty/white image
                            if not cropped_img.tobytes() == Image.new('RGB', (5, 5), color='white').tobytes():
                                
                                st.divider() # Add a divider to the Streamlit for better visual separation
                                st.subheader('') # Add some vertical space using an empty subheader
                                
                                # Use HTML to display a stylized header for color analysis
                                st.markdown("<h2 style='text-align: center;'>Color Analysis</h2>", unsafe_allow_html=True)
                                
                                st.subheader('') # Add more vertical space using an empty subheader
                                
                                cropped = cropped_img.convert('RGB') # Convert the cropped image to RGB format                            
                                image_bytes = io.BytesIO() # Create a BytesIO object to store the cropped image in memory
                                cropped.save(image_bytes, format='JPEG') # Save the cropped image to the BytesIO object in JPEG format
                                image_bytes.seek(0) # Reset the position of the BytesIO object
                                
                                # Initialize ColorThief with the in-memory image to extract dominant colors and a color palette
                                color_thief = ColorThief(image_bytes)
                                dominant_color = color_thief.get_color() # Extract the dominant color
                                color_palette = color_thief.get_palette(color_count=6) # Extract a color palette of 6 colors
                                
                                # Lists to store the extracted dominant color and color palette
                                dominant_color_list = []
                                color_palette_list = []
                                
                                # Append the colors to the respective lists
                                dominant_color_list.append(dominant_color)
                                color_palette_list.append(color_palette)
                                
                                # Define a function to convert an RGB tuple to its hex representation
                                def rgb_to_hex(rgb_tuple):
                                    r, g, b = rgb_tuple
                                    return "#{:02x}{:02x}{:02x}".format(r, g, b)            
                                
                                # Convert the dominant color to its hex representation
                                code = rgb_to_hex(dominant_color)
                                
                                # Split the Streamlit layout into two columns
                                col1,col2 = st.columns(2)
                                
                                # In the first column
                                with col1:
                                    # Display the cropped image
                                    st.image(cropped_img,use_column_width=True)                        
                                st.subheader('') # Add vertical spacing using an empty subheader
                                
                                # In the second column
                                with col2:
                                    st.title('') # Add vertical spacing using an empty title
                                    
                                    col1,col2 = st.columns(2) # Split the column into two more columns for better alignment
                                    
                                    # In the first sub-column
                                    with col1:
                                        # Use HTML to display a label for the dominant color
                                        st.markdown("<p style='padding:1.5em;font-size:20px;'>Dominant Color : </p>", unsafe_allow_html=True)
                                    
                                    # In the second sub-column
                                    with col2:
                                        st.title('') # Add vertical spacing using an empty title
                                        
                                        # Display the dominant color using a color picker widget, which will show the color but won't allow changes
                                        st.color_picker(label='dominant_color',value=code,label_visibility='collapsed')
                                    
                                    st.title('') # Add vertical spacing using an empty title
                                    
                                    # Use HTML to display a label for the color palette
                                    st.markdown("<p style='padding:1.5em;font-size:20px;'>Color Palette : </p>", unsafe_allow_html=True)
                                    
                                    st.title('') # Add vertical spacing using an empty title
                                    
                                    # Create seven columns to display the color palette
                                    columns = st.columns(7)
                                    
                                    # Iterate through the color palette and display each color using a color picker widget in the corresponding column
                                    for i, color in enumerate(color_palette):
                                        hex_color = rgb_to_hex(color)
                                        columns[i+1].color_picker(label=f"Color {i+1}", value=hex_color,label_visibility='collapsed')
                                
                                st.divider() # Add a divider to the Streamlit for better visual separation
                                
                                st.subheader('') # Add some vertical space using an empty subheader
                                
                                # Use HTML to display a stylized header for artworks with similar colors
                                st.markdown("<h2 style='text-align: center; color: black;'>Artworks with similiar colors</h2>", unsafe_allow_html=True)
                               
                                st.subheader('') # Add more vertical space using an empty subheader
                                
                                # Define a function to calculate the Euclidean distance between two RGB color values
                                def rgb_distance(rgb1, rgb2): 
                                    r1, g1, b1 = rgb1
                                    r2, g2, b2 = rgb2
                                    return ((r2 - r1) ** 2 + (g2 - g1) ** 2 + (b2 - b1) ** 2) ** 0.5
                                
                                # Define a function to find the closest matching color from a list of color names
                                def find_closest_color(rgb_color, color_list):
                                    closest_color = None
                                    min_distance = float('inf') # Initialize minimum distance to a high value
                                    closest_color_index = None
                                    
                                    # Iterate through the color list and find the closest matching color
                                    for i, color_name in enumerate(color_list):
                                        distance = rgb_distance(rgb_color, webcolors.name_to_rgb(color_name))
                                        if distance < min_distance:
                                            min_distance = distance
                                            closest_color = color_name
                                            closest_color_index = i
                                        
                                    return closest_color, closest_color_index
                                
                                # Extract the dominant RGB color from the list
                                rgb_color = dominant_color_list[0]
                                
                                # Define a list of color names for comparison
                                color_names = ['orangered', 'bisque', 'sandybrown', 'linen', 'antiquewhite', 'lavender', 'darkslateblue', 
                                                       'lightsteelblue', 'steelblue', 'midnightblue', 'cadetblue', 'wheat', 'goldenrod', 'palegoldenrod', 
                                                       'beige', 'khaki', 'rosybrown', 'indianred', 'maroon', 'darkolivegreen', 'darkkhaki', 'darkseagreen', 
                                                       'olivedrab', 'tan', 'sienna', 'peru', 'saddlebrown', 'burlywood', 'darkslategray', 'thistle', 'dimgray', 
                                                       'silver', 'gray', 'darkgray', 'lightgray', 'gainsboro', 'lightslategray', 'slategray', 'whitesmoke', 'palevioletred', 'black']
                                
                                # Find the closest color name to the dominant color using the defined function above
                                closest_color, closest_color_index = find_closest_color(rgb_color, color_names)
                                
                                # Define a caching decorator to ensure that the data is loaded only once and reused on subsequent calls
                                @st.cache_data
                                def final_v5():
                                    final_v5 = pd.read_csv(r"C:\streamlit_files\12_final_v5(0806).csv") # Load a CSV file into a Pandas DataFrame
                                    return final_v5
                                
                                simcol_df = final_v5() # Load the data using the defined function
                                selected_rows = simcol_df[simcol_df['rep_clr'] == closest_color] # Filter rows where the 'rep_clr' column matches the closest color
                                group = selected_rows.iloc[0]['group'] # Extract the 'group' value from the first row of the filtered DataFrame
                                selected_rows = simcol_df[simcol_df['web_cg_dt'] == group] # Filter rows based on the extracted 'group' value
                                random_sample = selected_rows.sample(n=9) # Randomly sample 9 rows from the filtered DataFrame
                                file_names = random_sample['file_name'].tolist() # Extract the 'file_name' values from the sampled rows into a list
                                
                                # Define paths to image folders of 12 art styles
                                folder_paths = [r"C:\streamlit_files\abstract_expressionism_img",
                                                        r"C:\streamlit_files\nap_img",
                                                        r"C:\streamlit_files\symbolism_img",
                                                        r"C:\streamlit_files\rc_img",
                                                        r"C:\streamlit_files\cu_img",
                                                        r"C:\streamlit_files\bq_img",
                                                        r"C:\streamlit_files\northern_renaissance_img",
                                                        r"C:\streamlit_files\impressionism_img",
                                                        r"C:\streamlit_files\romanticism_img",
                                                        r"C:\streamlit_files\sr_img",
                                                        r"C:\streamlit_files\expressionism_img",
                                                        r"C:\streamlit_files\realism_img"]
                                
                                # Define filename prefixes associated with 12 different art styles
                                files = ['abstract_expressionism_', 'nap_', 'symbolism_', 'rc_', 'cu_', 'bq_', 'orthern_renaissance',
                                                      'impressionism_', 'romanticism_', 'sr_', 'expressionism_', 'realism_']
                                
                                # Define a function to construct the full file path for a given art style and file number
                                def get_style_filename(prefix, number):
                                    idx = files.index(prefix) # Find the index of the provided prefix in the 'files' list
                                    folder_path = folder_paths[idx] # Extract the corresponding folder path
                                    filename = f'{prefix}{number}.jpg' # Construct the filename using the provided prefix and number
                                    file_path = os.path.join(folder_path, filename) # Construct the full file path by joining the folder path and filename
                                    return file_path
                                
                                numbers = file_names # Store the filenames in the 'numbers' variable
                                
                                # Create a new figure of specified size (10,10) for plotting
                                plt.figure(figsize=(10, 10))
                                
                                # Iterate over the filenames in 'numbers'
                                for i, num in enumerate(numbers):
                                    
                                    # Check for each prefix to determine the style and retrieve the correct image file
                                    for prefix in files:
                                        if num.startswith(prefix): # If the file name starts with the current prefix
                                            number = num[len(prefix):] # Extract the file number by removing the prefix
                                            file_path = get_style_filename(prefix, number) # Get the complete path using the function defined earlier
                                            image = imread(file_path) # Read the image from the file path
                                        
                                            plt.subplot(3, 3, i + 1) # Plot the image in a 3x3 grid
                                            plt.imshow(image) # Display the image
                                            plt.axis('off') # Hide axes ticks and labels
                                                    
                                            # Retrieve relevant information from the dataframe based on the filename
                                            row = simcol_df[simcol_df['file_name'] == num]
                                            if not row.empty: # Ensure that there's relevant data
                                                title = row['Title'].values[0] # Extract the title of the artwork
                                                painter = row['Painter'].values[0] # Extract the painter's name
                                                
                                                # Annotate the image with the title, wrapping the text for better readability
                                                plt.annotate(textwrap.fill(f"{title}", width=35), (0,0), (0, -10), xycoords='axes fraction', textcoords='offset points', va='top')
                                                
                                                # Calculate the y-offset for the painter's name based on the title's length
                                                n1 = (len(title)) // 35 
                                                if (len(title)) % 35 == 0:
                                                    n1 -= 1
                                                y1 = -23 - 13*n1
                                                
                                                # Annotate the image with the painter's name, wrapping the text for better readability
                                                plt.annotate(textwrap.fill(f"by {painter}", width=35), (0,0), (0, y1), xycoords='axes fraction', textcoords='offset points', va='top')
                                
                                plt.tight_layout(h_pad=5) # Adjust the layout for better display
                                st.pyplot(plt.gcf()) # Display the plotted figure in Streamlit
                                st.set_option('deprecation.showPyplotGlobalUse', False) # Suppress any warnings related to the use of pyplot in Streamlit
                                
                                st.divider() # Add a divider to Streamlit for visual separation
                                
                                st.subheader('') # Add some vertical space using an empty subheader            
                                
                                # Use HTML to display a stylized header indicating the artworks with similar styles
                                st.markdown("<h2 style='text-align: center; color: black;'>Artworks with similiar styles</h2>", unsafe_allow_html=True)
                                
                                # Define a caching decorator to ensure that the VGG model is loaded only once and reused on subsequent calls
                                @st.cache_resource
                                def vgg_model():
                                    model=load_model("C:/streamlit_files/vgg16.h5") # Load a pre-trained VGG16 model from the specified path
                                    return model
                                
                                m = vgg_model() # Load the VGG16 model using the defined function
                                x = img_to_array(cropped_img) # Convert the cropped image to an array format
                                
                                x = tf.image.resize(x, [224, 224]) # Resize the image to the size expected by the VGG16 model (224x224)
                                x = np.array([x]) # Expand the dimensions of the array to match the input shape expected by VGG16, i.e., (batch_size, height, width, channels)
                                predict = m.predict(pinp(x)) # Preprocess the resized image with pinp and then extract feature vectors using the VGG16 model
                                # from tensorflow.keras.applications.vgg16 import preprocess_input as pinp
                                
                                # Define a caching decorator to ensure that the total dataset is loaded only once and reused on subsequent calls
                                @st.cache_data
                                def total_db():
                                    # Load a serialized dataframe (pickle format) from the specified path
                                    file = open("C:/streamlit_files/total.txt","rb")
                                    total_df = pickle.load(file)
                                    file.close()
                                    return total_df 
                                
                                total=total_db() # Load the dataset using the defined function
                                index_predict = total['predict'] # Extract predicted feature vectors from the dataset; total['predict']: Values processed through VGG 16 feature extraction                            
                                similarities = [] # List to store similarity scores between the cropped image and the artworks in the dataset                                        
                                
                                # Calculate cosine similarities between the cropped image and all artworks in the dataset
                                for i in index_predict:
                                    similarities.append(cosine_similarity(predict, i))                                            
                                x = np.array(similarities).reshape(-1,) # Convert the list of similarity scores to a numpy array
                                # reshape the array to a one-dimensional form (a single vector) aka flatten                                           
                                
                                # Get the top 9 rows with the highest similarity scores and reset their indices
                                top_9 = total.iloc[np.argsort(x)[::-1][:9]].reset_index(drop=True)         
            
                                # Append a prefix to the 'url' column, indicating the location where the image files are stored                                   
                                top_9['url'] = top_9['url'].apply(lambda x: 'C:/streamlit_files/paintings/' + x)                                    
                                plt.figure(figsize=(10, 10)) # Initialize a new figure for plotting
                                i = 1 # Initial value for subplot indexing    
                                
                                # Iterate over the 'url' column of the top_9 DataFrame using enumerate, tracking index and url
                                for idx, url in enumerate(top_9['url']):
                                    image = imread(url) # # Read an image from the provided URL using imread
                                    plt.subplot(3, 3, i) # Create a subplot grid of 3x3 and select the i-th subplot (i starts from 1)
                                    plt.imshow(image) # Display the loaded image on the current subplot
                                    plt.axis('off') # Hide axes ticks and labels
                                    i += 1 # Increment the subplot counter i
                                    
                                    # Extract title and painter for the current artwork
                                    title = top_9['title'][idx]
                                    painter = top_9['painter'][idx]

                                    # Annotate the image with the artwork's title
                                    plt.annotate(textwrap.fill(f"{title}", width=35), (0,0), (0, -10), xycoords='axes fraction', textcoords='offset points', va='top')
                                    # (0,0) : x and y coordinates of the point where the annotation arrow will point to
                                    # In this case, (0,0) indicates that the arrow will point to the bottom-left corner of the plot
                                    
                                    # (0,-10) :  offset from the specified point (0,0) where the annotation text will be placed
                                    # positioned slightly above the point specified in the previous step
                                    
                                    # xycoords='axes fraction': specifies the coordinate system for the xy point, which is (0,0) in this case. 
                                    # 'axes fraction' means that the coordinates are given as fractions of the axes' width and height. 
                                    # In this case, (0,0) corresponds to the bottom-left corner of the plot
                                    
                                    # textcoords='offset points': specifies the coordinate system for the text's offset from the xy point
                                    # 'offset points' means that the offset is given in points (a unit of measurement in typography), which is commonly used in plotting
                                    
                                    # va='top': stands for "vertical alignment" and specifies how the annotation text is aligned with respect to the xy point. 
                                    # 'top' means that the top of the annotation text will be aligned with the xy point.
                                    
                                    # Calculate the number of lines needed for the title annotation
                                    n1 = (len(title)) // 35 # quotient
                                    if (len(title)) % 35 == 0: # remainder
                                        n1 -= 1
                                    y1 = -23 - 13*n1
                                    
                                    # Annotate the image with the painter's name
                                    plt.annotate(textwrap.fill(f"by {painter}", width=35), (0,0), (0, y1), xycoords='axes fraction', textcoords='offset points', va='top')
                        
                                plt.tight_layout(h_pad = 5) # Adjust the layout for better spacing between the plots
                                st.pyplot(plt.gcf()) # Display the plotted figure in the Streamlit
                                st.set_option('deprecation.showPyplotGlobalUse', False) # Suppress any warnings related to the use of pyplot in Streamlit / Disable the warning for deprecated use of pyplot in the context of st.pyplot
                        
                        # Catch any exceptions that might occur during the process
                        except:
                            st.subheader('')            
                
                # If an image was processed but not uploaded
                else:
                    st.subheader('You didnt upload your image')
            
            # If the user didn't upload an image
            else:
                st.write("Please upload an image")

# When the selected option is 'Neural Style Transfer'
elif selected=='Neural Style Transfer':
    
    # Set the title for the page
    st.title('Neural Style Transfer')
    st.header('') # Add an empty header for spacing
    
    # Create two columns for uploading original and style images
    col1, col2 = st.columns(2)
    
    # In the first column, allow users to upload an original image
    with col1:
        original_image = st.file_uploader(label='Choose an original image', type=['jpg', 'jpeg'])
        
        if original_image : 
            # Display the uploaded original image
            st.image(image=original_image,
                     caption='Original Image',
                     use_column_width=True)
    
    # In the second column, allow users to upload a style image
    with col2: 
        style_image = st.file_uploader(label='Choose a style image', type=['jpg', 'jpeg'])
        
        if style_image :
            
            # Display the uploaded style image    
            st.image(image=style_image,
                         caption='Style Image',
                         use_column_width=True)    
    
    st.header('') # Add another empty header for spacing
    
    button=None # Initialize a button variable
    
    # Define a function to load and preprocess an image
    def load_image(image_file, image_size=(512, 256)):
        content = image_file.read() # read the content of the uploaded image file
        
        # Decode the image content using TensorFlow's decode_image function 
        # Convert the image to a tensor with an additional dimension using [tf.newaxis, ...]
        img = tf.io.decode_image(content, channels=3, dtype=tf.float32)[tf.newaxis, ...]
        
        # Resize the image while preserving its aspect ratio
        img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
        return img
    
    # If both original and style images are uploaded
    if original_image and style_image :
        
        # Create columns for layout
        col1,col2,col3,col4,col5 = st.columns(5)
        with col3 : 
            
            # Create a stylize button
            button = st.button('Stylize Image')
            
            if button :
                
                # Display a spinner while processing
                with st.spinner('Running...') :
                    
                    # Load and preprocess the images using the load_image function defined above
                    original_image = load_image(original_image)
                    style_image = load_image(style_image)
                    
                    # Preprocess the style image
                    style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='VALID')
                   
                   # Load the arbitrary image stylization model 
                    @st.cache_resource
                    def ais():
                        ais_model=tf_hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
                        return ais_model
                    stylize_model = ais()
            
                    # Stylize the image using the loaded model
                    results = stylize_model(tf.constant(original_image), tf.constant(style_image))
                    stylized_photo = results[0].numpy().squeeze()  # Convert to NumPy array and squeeze dimensions
                    stylized_photo_pil = PIL.Image.fromarray(np.uint8(stylized_photo * 255))  # Convert to PIL image and rescale to [0, 255]
    
    st.header('') # Add an empty header for spacing
    
    col1,col2,col3 = st.columns(3)
    
    # If the stylize button was pressed
    if button :
        with col2 :
            # Display the stylized image
            st.image(image=stylized_photo_pil,
                             caption='Stylized Image',
                             use_column_width=True)
            
            # Add a rain animation using a custom rain function
            rain(
                emoji="🎈",
                font_size=30,
                falling_speed=10,
                animation_length="infinite"
                )

# When the selected menu option, if it's 'Artwork MBTI', proceed
elif selected == 'Artwork MBTI':

    # Function to resize the image to a given width and height
    def resize_image(image, width, height):
        return image.resize((width, height), Image.Resampling.LANCZOS)

    # Function to run the sequential matchup game to determine user's MBTI based on art style preference
    def sequential_matchup_game(images, image_folder, mbti_data):
        st.subheader("더 마음에 드는 사진을 골라주세요 :smile:")
        # Instructions for the user
        st.write("본 미니게임은 11라운드로 진행되는 토너먼트식 게임입니다.")

        # Pair up image indices with their respective images
        image_list = list(zip(range(len(images)), images))

        # Initialize the match counter
        match_count = 0

        # Set the dimensions for displaying images
        width, height = 750, 572

        # Continue matchups until there's only one image left
        while len(image_list) > 1:
            match_count += 1
            st.write(f"{match_count}번째 라운드 :point_down: ")
            
            # Display two competing images side by side
            col1, col2 = st.columns(2)
            image_1 = image_list[0]
            image_2 = image_list[1]

            # Display the first image
            with col1:
                st.image(resize_image(image_1[1], width, height), use_column_width=True, caption='첫번째 이미지')

            # Display the second image
            with col2:
                st.image(resize_image(image_2[1], width, height), use_column_width=True, caption='두번째 이미지')

            # Let the user choose their preferred image
            choice = st.radio(f"어느 쪽이 더 좋나요? {match_count} 번째 선택", ('선택안함', '첫번째 이미지', '두번째 이미지'))

            # Process the user's choice
            if choice == '선택안함':
                # If the user doesn't make a choice, prompt them to select an option
                st.write("선택을 진행해주세요. 당신의 MBTI 유형을 맞혀보겠습니다. :bulb:")
                break
            
            elif choice == '첫번째 이미지':
                # If the user chooses the first image, move it to the end of the list and remove the first two images
                image_list.append(image_1)
                image_list.pop(0)
                image_list.pop(0)
            
            elif choice == '두번째 이미지':
                # If the user chooses the second image, move it to the end of the list and remove the first two images
                image_list.append(image_2)
                image_list.pop(0)
                image_list.pop(0)

            # Inform the user of the next step or the end of the game
            if match_count != 11:
                st.info('선택을 마쳤습니다. 스크롤을 내려 다음 라운드를 진행해주세요.', icon="ℹ️")
            else:
                st.info('모든 라운드가 끝났습니다. 스크롤을 내려 결과를 확인해주세요.', icon="ℹ️")

        # Once the matchups are done, display the winning image
        if len(image_list) == 1:
            winner_image = image_list[0]
            st.subheader("경기 종료!")
            st.write("최종 선택을 받은 작품은 :")
            st.image(resize_image(winner_image[1], width, height), use_column_width=True)

            # Fetch the MBTI data for the winning image
            mt = mbti_data.iloc[winner_image[0]]
            mbti_exp_info = mt['exp']
            mbti_short = mt['mbti']
            mbti_style = mt['style']
            st.subheader(mbti_style + " 작품이 제일 마음에 드는 당신의 MBTI 유형은....")
            st.subheader(mbti_short + ' 입니까:question:')
            st.write(mbti_exp_info)
    
    # The main function that initiates the entire MBTI art style test
    def main():
        st.title("Mini Game - 미술사조 mbti test :heart:") # Set the page title
        image_folder = "C:/streamlit_files/mbti/"  # Define the directory where the images are stored
        
        # List comprehension to generate the image filenames
        # It assumes there are 12 images named from img_1.jpg to img_12.jpg
        image_names = [f"img_{i}.jpg" for i in range(1, 13)]  
    
        images = [Image.open(image_folder + name) for name in image_names] # Load all images from the folder into a list
    
        mbti_data = pd.read_csv(r"C:\streamlit_files\style_mbti_v2.csv") # Load the MBTI data from a CSV file
        
        # Call the sequential_matchup_game function with the loaded images and MBTI data
        sequential_matchup_game(images, image_folder, mbti_data) 
    
    # This condition ensures that the main() function is only called if the script is being run directly and not imported as a module in another script
    if __name__ == "__main__":
        main()