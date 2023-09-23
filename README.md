# Know Thy Art Part One 
Group Project for Machine Learning Bootcamp

# Purpose 

In a world where art appreciation often remains passive and surface-level for many, our project aims to transform and deepen users' engagement with art. We've recognized that art isn't just about observation, but about connection, immersion, and personal resonance. With this understanding, we've crafted a suite of tools and services designed to illuminate, engage, and inspire :

-  **Western Art Style Insight Tool** : Leveraged ResnetRS50 model to create an art style classifier. This tool serves not only as an identifier but also as an enlightener, guiding users on the nuances of each art style. In doing so, it elevates daily encounters with art from passive observation to a profound understanding.
-  **Artwork Recommendation Engine** : Developed artwork recommendation service based on color and style. With this, we allow users to deepen their understanding of art through exploring more related artworks
-  **Neural Style Transfer** : Beyond just appreciation, this service allow users to craft their own masterpieces by blending distinct art styles. This goes beyond conventional tools — it's a personal canvas and an invitation to partake in the art creation journey, kindling a newfound passion and personal connection to the realm of art
- **Artwork MBTI Predictor** : A confluence of personality and art, this service is our endeavor to align art preferences with personality types, fostering a more intuitive and personal connection between viewers and artworks

With these services, we're not just showcasing art; we're crafting an enriched, personalized, and deeply engaging art experience for everyone

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

**7 Art Styles Chosen For Classification** : 
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

# Crop Painting Using Object Detection 
![image](https://github.com/ryan-hk-koo/know_thy_art_part_1/assets/143580734/25891fee-b743-4323-85ad-297079c6a492)
- Used YOLOv8m for painting detection in a photo or an image
- Trained the model with 560 training samples and 140 validation samples
- Achieved an average precision of 0.994 


