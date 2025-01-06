# Data-Science-Project-Galaxy-Morphology-Classification-Using-Deep-Learning-and-Explainable-AI-

This repository contains code for training, evaluating, and interpreting deep learning models for galaxy classification using the **Galaxy Zoo: The Galaxy Challenge** dataset. The focus is on classifying galaxies into **Spiral** and **Elliptical** types using models such as **VGG16**, **ResNet50**, and a **Custom Convolutional Neural Network (CNN)**. The project emphasizes accuracy, explainability, and effective visualization of results.

---

## **Features**
- Preprocessing and filtering of Galaxy Zoo dataset.
- Exploratory Data Analysis (EDA) to visualize class distributions and image properties.
- Implementation of data augmentation for robust training.
- Model training using transfer learning with **VGG16** and **ResNet50**, as well as a custom CNN.
- Model evaluation with metrics such as accuracy, precision, recall, F1-score, and ROC-AUC curves.
- Explainability methods, including **Grad-CAM** and **LIME**, to understand model predictions.

---

## **Setup and Requirements**

## **Required Python Libraries**
Ensure the following libraries are installed to run the project:

- `numpy`: For numerical computations.
- `pandas`: For data manipulation and analysis.
- `matplotlib`: For creating visualizations.
- `seaborn`: For advanced visualizations.
- `tensorflow`: For building and training neural networks.
- `scikit-learn`: For preprocessing and evaluation metrics.
- `opencv-python`: For image processing.
- `Pillow`: For image loading and manipulation.
- `lime`: For explainable AI methods.
- `imbalanced-learn`: For handling imbalanced datasets.

Install missing libraries using `pip`. For example:
```bash
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn opencv-python Pillow lime imbalanced-learn
```
## **Dataset**

The dataset used in this project is sourced from the **Galaxy Zoo: The Galaxy Challenge**, which provides galaxy images and corresponding classification labels.

### **Steps to Obtain and Prepare the Dataset**

1. **Download the Dataset**:
   - Visit the [Galaxy Zoo Challenge dataset page](https://data.galaxyzoo.org/) and download the required files.

2. **Dataset Contents**:
   - **Images**: High-resolution galaxy images provided in the folder `images_training_rev1/`.
   - **Labels**: Classification probabilities for each galaxy provided in the file `training_solutions_rev1.csv`.

3. **Dataset Setup**:
   - Place the images in the directory: `images_training_rev1/`.
   - Place the labels file in the project root or specify its path in the code: `training_solutions_rev1.csv`.

4. **Structure**:
   - Each galaxy is associated with a unique `GalaxyID` and corresponding classification probabilities for various categories (e.g., Spiral, Elliptical, Irregular).
   - Example of label data:
     ```
     GalaxyID,Class1.1,Class1.2,...,Class8.5
     100008,0.1,0.9,...,0.0
     100023,0.85,0.05,...,0.0
     ```

5. **Preprocessing**:
   - Filter the dataset to include only **Spiral** and **Elliptical** galaxies based on thresholds:
     - `Class1.2` ≥ 0.9 → Spiral
     - `Class1.1` ≥ 0.85 → Elliptical
   - Exclude galaxies not meeting the thresholds or falling into ambiguous categories.
   - Encode the labels into binary classes (`Spiral` and `Elliptical`) for model training.

6. **Splitting**:
   - The dataset is split into **Training**, **Validation**, and **Testing** sets with a stratified sampling approach:
     - **Training**: 70% of the filtered dataset.
     - **Validation**: 15% of the filtered dataset.
     - **Testing**: 15% of the filtered dataset.

By following these steps, the dataset is prepared for training deep learning models. Ensure that the paths for the images and labels are correctly specified in the code.

## **Project Structure**

The project pipeline is structured into several key stages to ensure clarity and reproducibility. Below is an overview of each stage:

### **1. Preprocessing**
- **Objective**: Prepare the dataset for model training.
- Steps:
  1. Load and inspect the Galaxy Zoo dataset (images and labels).
  2. Filter the dataset to focus on **Spiral** and **Elliptical** galaxies:
     - Spiral galaxies: `Class1.2 ≥ 0.9`
     - Elliptical galaxies: `Class1.1 ≥ 0.85`
  3. Encode the galaxy types into binary classes: `Spiral` and `Elliptical`.
  4. Split the dataset into training, validation, and test sets (70%-15%-15% split).
  5. Perform data augmentation for the training set using `ImageDataGenerator` to enhance model robustness.

---

### **2. Exploratory Data Analysis (EDA)**
- **Objective**: Understand the dataset distribution and characteristics.
- Key Analyses:
  - **Class Distributions**:
    - Visualize the number of galaxies in each category (Spiral vs. Elliptical).
  - **Image Properties**:
    - Analyze brightness levels, pixel means, and standard deviations across the dataset.
  - **Sample Images**:
    - Display example images for each galaxy type.

---

### **3. Model Training**
- **Objective**: Train deep learning models for galaxy classification.
- Models:
  1. **VGG16**:
     - Pre-trained on ImageNet.
     - Fine-tuned dense layers added for binary classification.
  2. **ResNet50**:
     - Pre-trained on ImageNet.
     - Modified for galaxy classification with additional dense layers.
  3. **Custom CNN**:
     - A fully custom-built convolutional neural network designed from scratch.

---

### **4. Evaluation**
- **Objective**: Assess model performance.
- Metrics:
  - Accuracy, Precision, Recall, and F1-score.
  - Confusion matrix to evaluate class-wise performance.
  - ROC-AUC curves to compare model effectiveness.

---

### **5. Explainable AI**
- **Objective**: Interpret model predictions using explainability techniques.
- Methods:
  1. **Grad-CAM**:
     - Highlight regions in galaxy images that influence the model's classification decision.
  2. **LIME**:
     - Provide superpixel-based explanations for local predictions.

---

### **6. Results and Comparison**
- **Objective**: Compare the performance of VGG16, ResNet50, and Custom CNN.
- Key Visualizations:
  - ROC-AUC curves for all models.
  - Classification reports summarizing Precision, Recall, and F1-scores.
  - Confusion matrices for detailed error analysis.

---

### **7. Future Enhancements**
- Suggestions for improving the project:
  - Explore other architectures like **EfficientNet**.
  - Implement model ensembling for better accuracy.
  - Extend the dataset to include additional galaxy categories.

## **Results**

The performance of the models—**VGG16**, **ResNet50**, and **Custom CNN**—was evaluated using metrics such as precision, recall, F1-score, and accuracy. Below is a summary of the results:

### **Model Performance Metrics**

| **Model**     | **Metric**   | **Elliptical** | **Spiral** | **Accuracy** |
|---------------|--------------|----------------|------------|--------------|
| **VGG16**     | Precision    | **1.00**       | **0.98**   | **0.99**     |
|               | Recall       | **0.98**       | **1.00**   |              |
|               | F1-score     | **0.99**       | **0.99**   |              |
| **Custom CNN**| Precision    | 0.71           | 0.64       | 0.67         |
|               | Recall       | 0.57           | 0.76       |              |
|               | F1-score     | 0.63           | 0.70       |              |
| **ResNet50**  | Precision    | 0.94           | 0.90       | 0.92         |
|               | Recall       | 0.90           | 0.94       |              |
|               | F1-score     | 0.92           | 0.92       |              |

### **Observations**

1. **VGG16**:
   - Achieved the **highest accuracy (99%)** among all models.
   - Perfect precision for **Elliptical galaxies** and near-perfect for **Spiral galaxies**.
   - Balanced recall and F1-scores across both classes.

2. **Custom CNN**:
   - Showed lower performance compared to pre-trained models.
   - Precision, recall, and F1-scores indicate that the model struggles to generalize, particularly for **Elliptical galaxies** (Recall = 0.57).

3. **ResNet50**:
   - Delivered solid performance with **92% accuracy**.
   - Precision and recall metrics were well-balanced for both classes.
   - A viable alternative to VGG16 for scenarios requiring lower computational overhead.

---

### **Visual Results**

1. **Confusion Matrices**:
   - Visualized the true vs. predicted classifications for each model.
   - Highlighted areas where the models struggled (e.g., misclassifications).

2. **ROC-AUC Curves**:
   - Compared the ability of models to distinguish between the two classes.
   - **VGG16** had the highest AUC score, closely followed by **ResNet50**.

3. **Explainable AI**:
   - **Grad-CAM** heatmaps illustrated which regions of galaxy images influenced model decisions.
   - **LIME** provided superpixel-based local interpretability for individual predictions.

---

### **Key Insights**
- **VGG16** is the most suitable model for deployment, achieving the best overall performance.
- **ResNet50** provides a balanced alternative with slightly lower accuracy.
- **Custom CNN** requires additional improvements, such as deeper layers or pretraining on a related dataset, to compete with transfer learning models.

---

This section summarizes the quantitative and qualitative outcomes of the project, making it easy for readers to understand the comparative performance of the models.

