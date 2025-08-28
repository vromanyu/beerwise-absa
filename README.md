# Aspect-Based Sentiment Analysis on Beer Reviews

This project is my thesis on Aspect-Based Sentiment Analysis (ABSA), focusing on beer reviews. Each review in our dataset includes scores for four aspects: **appearance**, **aroma**, **palate**, and **taste**.


## Data Preparation
- **Dataset Storage:** The original BeerAdvocate dataset and the single filtered/processed dataset are stored under the `dataset` folder. Due to their size, these files are tracked using **git LFS** (Large File Storage). To download these files after cloning the repository, run:

	```bash
	git lfs pull
	```
- **Preprocessing:** All reviews were preprocessed (tokenization, cleaning, etc.) to ensure high-quality input for modeling.
- **Aspect Identification:** Using the FastText model, we identified which aspects were mentioned in each review. Since each review had scores for appearance, aroma, palate, and taste, we assumed these were the aspects discussed.
- **Aspect Pair Selection:** We found that the most frequently mentioned aspect pair was **(appearance, palate)**. We filtered the dataset to include only reviews where both aspects were clearly mentioned.

## Modeling Approaches

### Logistic Regression Model
- **Upsampling & Balancing:** We created a joint label combining appearance and palate sentiments. The dataset was upsampled so each joint label class had equal representation, ensuring balanced training.
- **Splitting:** The data was split into train and test sets using stratified sampling on the joint label, preserving class distribution.
- **Feature Extraction:** TF-IDF vectorization was used, with extensive tuning of `max_features` and `ngram_range`.
- **Multi-Output Classification:** A multi-output logistic regression model was trained to predict sentiment for both aspects simultaneously.

### Linear SVC Model
- The pipeline for the Linear SVC model is similar to the Logistic Regression approach, including upsampling, stratified splitting, TF-IDF vectorization, and multi-output classification.
- The main difference is the use of LinearSVC as the base estimator, with class weights for handling class imbalance and specific hyperparameters for robust sentiment prediction.

### Ridge Classifier Model
- The pipeline for the Ridge Classifier model is similar to the Logistic Regression and Linear SVC approaches, including upsampling, stratified splitting, TF-IDF vectorization, and multi-output classification.
- The main difference is the use of RidgeClassifier as the base estimator, which applies L2 regularization and can handle class imbalance using class weights.
- Ridge Classifier is robust to multicollinearity and can provide good separation of classes, but may not always outperform Logistic Regression or Linear SVC for text data.

### Multinomial Naive Bayes Model
- The pipeline for the Multinomial Naive Bayes model is similar to the Logistic Regression and Linear SVC approaches, including upsampling, stratified splitting, TF-IDF vectorization, and multi-output classification.
- The main difference is the use of MultinomialNB as the base estimator, which is fast to train and simple.
- **Limitations:**
  - Naive Bayes assumes all features are independent given the class label, which is rarely true for text data.
  - It relies on simple probability estimates, which can limit its effectiveness compared to more flexible models.
  - Logistic Regression and Linear SVC can model more complex relationships and interactions between features, leading to better separation of classes.
  - Does not support class weights for handling class imbalance.

### Transformer-Based Model
- **Upsampling & Balancing:** Similar to the logistic regression pipeline, joint labels were created and upsampled for balance.
- **Splitting:** Stratified splits ensured balanced train, validation, and test sets.
- **Modeling:** A transformer-based neural network (e.g., BERT) was trained to predict aspect sentiments, using custom loss functions and class weights for further balancing.
- **Hardware Acceleration:** PyTorch was used with the CUDA version to accelerate training of transformer-based models. **CUDA is necessary to train transformer-based models in this project, since the implementation uses CUDA-specific features such as `GradScaler`.** To load the trained models CUDA is not needed.  
  **Note:** The PyTorch requirement is commented out in `requirements.txt`. Please install the appropriate PyTorch version manually:
  - If you have a CUDA GPU, install the CUDA version of PyTorch.
  - If you do not have CUDA, install the CPU version as follows:

  ```bash
  # CPU version
  pip3 install torch torchvision 
  
  # GPU version (example for CUDA 12.9)
  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129

  ```

## Results
- **Logistic Regression and Linear SVC:** Achieved strong macro F1-scores and balanced performance across all sentiment classes, thanks to upsampling, stratified splitting, and TF-IDF feature extraction. These models effectively captured complex relationships in the data, resulting in good class separation.
- **Ridge Classifier:** Provided competitive results, with the added benefits of robustness to multicollinearity and the ability to handle class imbalance. However, its performance was generally in line with Logistic Regression and Linear SVC, without significant advantages.
- **Multinomial Naive Bayes:** Trained very quickly and provided reasonable baseline results, but macro F1-scores were generally lower than those of Logistic Regression and Linear SVC. Limitations in modeling feature interactions and handling class imbalance were reflected in the results.
- **Transformer-Based Model:** Delivered the highest macro F1-scores and most robust performance, especially for minority classes. Training was accelerated using PyTorch with CUDA, and the model required more computational resources. Custom loss functions and class weights further improved class balance.
- **General Observation:** Proper preprocessing, aspect identification, and data balancing were critical for achieving balanced and high-performing models across all approaches.

---

**Keywords:** Aspect-Based Sentiment Analysis, Beer Reviews, FastText, Logistic Regression, Transformer, Data Balancing, Multi-Output Classification
