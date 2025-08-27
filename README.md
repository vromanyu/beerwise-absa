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

### Transformer-Based Model
- **Upsampling & Balancing:** Similar to the logistic regression pipeline, joint labels were created and upsampled for balance.
- **Splitting:** Stratified splits ensured balanced train, validation, and test sets.
- **Modeling:** A transformer-based neural network (e.g., BERT) was trained to predict aspect sentiments, using custom loss functions and class weights for further balancing.

## Results
Both approaches achieved high macro F1-scores and balanced performance across all sentiment classes, demonstrating the effectiveness of proper preprocessing, aspect identification, and data balancing techniques.

---

**Keywords:** Aspect-Based Sentiment Analysis, Beer Reviews, FastText, Logistic Regression, Transformer, Data Balancing, Multi-Output Classification
