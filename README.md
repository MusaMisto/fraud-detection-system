# Advanced AI-Driven Fraud Detection System for Transactional Security

**Project Overview**  
This project implements an **Advanced AI-Driven Fraud Detection System** designed for **transactional security**. The system utilizes machine learning algorithms, including **Random Forest** and **Logistic Regression**, to detect fraudulent transactions with high accuracy. The solution is optimized using **SMOTE** for handling class imbalances, ensuring robust performance in real-world scenarios.

---

## Features

- **Preprocessing Pipeline:**  
  - Handles missing data and categorical encoding.
  - Implements feature engineering techniques like creating time-based features.
  - Scales numerical data using **MinMaxScaler**.

- **Model Training and Evaluation:**  
  - Includes multiple classifiers such as **Random Forest**, **Gradient Boosting**, **Logistic Regression**, **Naive Bayes**, and **Decision Tree**.
  - Utilizes **SMOTE** and **Random Undersampling** for class imbalance management.
  - Provides detailed evaluation metrics including **F1 Score**, **AUC-ROC**, **Recall**, **Accuracy**, and **Precision**.

- **Visualizations:**  
  - Presents data distributions and feature importances through visually appealing plots.
  - Generates performance comparison charts for different models.

- **Deployment Ready:**  
  - Contains a **FastAPI** application for real-time transaction fraud detection.
  - Includes endpoints for prediction and a web interface for easy interaction.

---

## Installation

To install and set up this project, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MusaMisto/fraud-detection.git
   ```

2. **Navigate to the project directory:**
   ```bash
   cd fraud-detection
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the FastAPI server:**
   ```bash
   uvicorn app.main:app --reload
   ```

---

## Usage

- **Model Training:**
  - To train the model, run the **.py** script or use the provided **.ipynb** notebook.
  - Adjust hyperparameters and algorithms as needed in the code.

- **Web Interface:**
  - The FastAPI web interface allows you to upload transaction data and receive real-time fraud predictions.
  - The results are displayed directly on the web page, with visual feedback (green/red background) for legitimacy or fraud.

---

## License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute this software, provided you include proper attribution.

---

## Contributing

Contributions are welcome! If you want to contribute to this project, please open an issue or submit a pull request on GitHub.

---

## Contact

For more information or inquiries, please contact **[Musa Misto](mailto:misto.musa02@gmail.com)**.

---

## Acknowledgements

This project was developed during an internship at **[Optimiza]**. Special thanks to the team for their support and guidance.
