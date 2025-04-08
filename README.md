# NJIT-CS375-AfsheenK
**Intel Image Classification Dataset**: [Kaggle Dataset Link](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

Description: This dataset was created for the purpose of image classification, specifically for natural scenes. It contains images categorized into six classes: buildings, forest, glacier, mountain, sea, and street. The images were collected from various online sources. The dataset contains approximately 25,000 images, with roughly 2,500 images per class. Key features include images that are RGB (color) with varying resolutions. The primary feature is the pixel data representing the visual content of the scenes.

## Classification Problem:

**Problem**: Accurately classify images into the six scene categories (buildings, forest, glacier, mountain, sea, street).

**Input Features**: Pixel data from the images.

**Target Variable**: The scene category (categorical).

**Potential Challenges**: Variations in lighting, weather, and camera angles; potential class imbalance; computational demands of image processing.

**Real-World Relevance**: Environmental monitoring, urban planning, autonomous vehicle navigation, geographic information systems.

## Regression Problem:

**Problem:** Estimate the "environmental impact score" of a given scene. For example, a higher score can be assigned to scenes with more human-made structures (buildings, streets) and a lower score to natural scenes (forests, glaciers).

**Input Features:** The classified scene category (from the first problem) and potentially supplemental environmental data.

**Target Variable:** A numerical score representing the estimated environmental impact.

**Potential Challenges:** Defining an objective "environmental impact score," correlating visual scenes with actual environmental data.

### Real-World Relevance:
Environmental impact assessment, sustainability studies, policy making.

### Reflection: 
This project allows for a deep dive into image classification using CNNs while exploring the connection between visual scenes and environmental impact. It encourages critical thinking about how image analysis can contribute to understanding our environment.

Ethical Considerations: Potential biases in scene classification (e.g., misrepresenting certain environments), the subjective nature of "environmental impact."
