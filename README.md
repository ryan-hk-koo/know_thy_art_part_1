# Know Thy Art Part One 

<br>

- Streamlit Demo on Youtube

[![Streamlit Demo](http://img.youtube.com/vi/r_SdumnMuB8/0.jpg)](http://www.youtube.com/watch?v=r_SdumnMuB8 "Streamlit Demo")

                          
# Purpose 

While many often engage with art on a superficial level, our project seeks to transform and deepen users' engagement with art. To achieve this, we have crafted the following suite of services:

- **Know Thy Art**:
  -  **Western Art Style Insight Tool**: Leveraged ResnetRS50 model to create an art style classifier. This tool not only identifies art styles but also educates users on the nuances of each style. In doing so, it elevates daily encounters with art from passive observation to a profound understanding
  -  **Artwork Recommendation Engine**: Developed artwork recommendation services based on RGB colors and style. With this, we allow users to deepen their understanding of art through exploring more related artworks
-  **Neural Style Transfer**: This tool lets users create their own unique artworks by blending different art styles. It offers a new way to interact with and create art, promoting a deeper personal connection
- **Artwork MBTI Predictor**: This tool matches art preferences with personality types, helping users find artworks that resonate with their personality and fostering a more intuitive and personal connection between viewers and artworks

<br>

# Dataset
- Collected approximately 3,600 data entries (images, artwork information) for each of the 15 art styles from [WikiArt](https://www.wikiart.org/)
<br>

![image](https://github.com/ryan-hk-koo/know_thy_art_part_1/assets/143580734/e0ed6a6f-8b74-4f72-80e4-f8fd208e9cbe)


**Selection Criteria for Art Styles** :
1. Prioritized based on data volume, assuming a larger dataset leads to better model training
2. Chose one among similar styles (e.g., selected Impressionism over Post-Impressionism)
3. Excluded styles with a limited number of paintings, such as Art Nouveau and Neoclassicism
4. Of the remaining 12 styles, picked the 7 that demonstrated the highest model classification accuracy

**7 Art Styles Chosen for Classification** : 
- Abstract Expressionism, Baroque, Cubism, Impressionism, Primitivism, Rococo, Surrealism

<br>

# Data Preprocessing
![image](https://github.com/ryan-hk-koo/know_thy_art_part_1/assets/143580734/44fb7b34-42d1-40a8-9b5c-33909f3616b7)
- Excluded multi-style paintings, sketches, sculptures, illustrations, as well as black and white or non-rectangular paintings
- Cropped out frames and eliminated blank spaces within images

<br>

# Data Preprocessing Result 
![image](https://github.com/ryan-hk-koo/know_thy_art_part_1/assets/143580734/e59c1fb8-cc7d-44f4-9f20-d68eedd48566)
- Initial set of 25,055 images reduced to 17,924 images after preprocessing

<br> 

# Automatic Painting Extraction with Object Detection
![image](https://github.com/ryan-hk-koo/know_thy_art_part_1/assets/143580734/25891fee-b743-4323-85ad-297079c6a492)
- Implemented an object detection system tailored for paintings using YOLOv8m
  - Defined 'Painting' as the sole detection class
  - Model was trained on 560 images and validated on 140 images
  - Attained an average precision of 0.994
- When a user uploads an image, our system identifies and isolates the artwork, excluding any extraneous backgrounds or frames
- This ensures that only the actual artwork is considered, optimizing the subsequent style classification process and minimizing potential biases from external elements (i.e. backgrounds and frames)

![image](https://github.com/ryan-hk-koo/know_thy_art_part_1/assets/143580734/539a9bda-27cc-4ee9-84ca-baf04b6567ee)
- As demonstrated in the third example, the model was also trained on paintings without any surrounding context. This ensures that full images aren't cropped when they depict artworks alone
<br>

# CNN Model Selection 
![image](https://github.com/ryan-hk-koo/know_thy_art_part_1/assets/143580734/b9c596cb-4c3e-48e0-b956-996c0433d1d1)

- For CNN model selection, we used 500 sample data points for each art style:
  - 400 for training
  - 50 for validation
  - 50 for testing
- A total of 18 models were tested, as outlined in the table above
- ResNetRS50 achieved the highest test accuracy with 65.83%, hence ResNetRS50 was selected as the classification model 

<br>

# Model Fine Tuning 
![image](https://github.com/ryan-hk-koo/know_thy_art_part_1/assets/143580734/28eb7c02-17cb-4a67-82fd-addfde6d3891)
- Selected a dense layer with 10,240 units by iteratively doubling its size until there was no further increase in accuracy

<br>

# Model Results 

![image](https://github.com/ryan-hk-koo/know_thy_art_part_1/assets/143580734/80d1302b-67ed-47e7-bf31-f8b067868fef)
- ResnetRS50 without data augmentation outperformed its counterpart that utilized data augmentation, registering a 4% increase in accuracy with test data
- The model without data augmentation consistently demonstrated greater confidence in its top-1 accuracy predictions
- Hence, selected the ResnetRS50 without data augmentation as the primary model for the art style classification service

<br>

# Art Recommender by RGB colors 

**Preparation**:
- Utilized ColorThief library to extract the RGB values of the dominant colors and color palettes from the image dataset
- Employed the WebColors library to identify the names of the dominant colors, or their closest matches, from the 138 CSS3 colors based on its RGB values
- Grouped 138 colors from WebColors library into 41 color groups based on color frequencies and similarities
  - 3 Orange, 3 White, 5 Blue, 5 Yellow, 3 Red, 4 Green, 5 Brown, 2 Purple, 9 Gray, 1 Pink, 1 Black Groups
 
**User Experience**:
- When a user inputs an image, the system first extracts the RGB values of the dominant color
- The RGB values are then matched to the closest color name among the 138 CSS3 colors, based on Euclidean distance
- Once the closest color name is identified, the system determines which of the 41 groups the color belongs to and displays the images from that group
- Example:

![image](https://github.com/ryan-hk-koo/know_thy_art_part_1/assets/143580734/163c71d6-5649-421a-bf7b-27a00b26e94a)

<br>

# Art Recommender by Drawing Style 
![image](https://github.com/ryan-hk-koo/know_thy_art_part_1/assets/143580734/6c6d8f5f-8871-470c-b8ba-c1157bea65f1)

<br>

![image](https://github.com/ryan-hk-koo/know_thy_art_part_1/assets/143580734/40df1d6b-8d4d-4665-a7e5-2fdc52ba6f2e)

<br>

![image](https://github.com/ryan-hk-koo/know_thy_art_part_1/assets/143580734/d06e8f0b-7551-4174-97fb-4ac861fb4bb8)

<br>

![image](https://github.com/ryan-hk-koo/know_thy_art_part_1/assets/143580734/e6a16ac5-9cc2-466f-84d3-8eccbb0173ce)

<br>

# Neural Style Transfer
- The code is sourced from [this article](https://towardsdatascience.com/python-for-art-fast-neural-style-transfer-using-tensorflow-2-d5e7662061be) on Towards Data Science

<br>

# Artwork MBTI Predictor
- The correlation between art style preference and MBTI is sourced from [this website](https://personalitylist.com/category/generic/visual-art-genres/)

<br>

# Conclusion & Reflections
- Our model successfully classified seven distinct art styles with an accuracy of 84%, underscoring the discernible differences in these styles that a convolutional neural network can pinpoint and categorize
- Interestingly, the model that did not utilize data augmentation outperformed its augmented counterpart by 4%
  - This suggests that either data augmentation might not be beneficial for classifying art styles, or the specific augmentation techniques we employed were not optimal
- Preprocessing images proved crucial to the model's performance. Initially, without preprocessing, the model's accuracy was only 50%
- Fine-tuning the preprocessing criteria could further elevate the accuracy of the CNN model
- Depending on the objectives, future projects on art recommendation by color could consider various methods such as 3D HSV that take into account both brightness and saturation or k-means clustering
- In finding images with similar drawing styles, there is a need for more objective numerical metrics to compare the performances of the models









