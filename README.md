# House Price Prediction with Linear Regression

A machine learning project that predicts residential house prices using **Linear Regression** and the Ames Housing dataset. The project demonstrates a complete baseline regression workflow, from data preprocessing and categorical encoding to model evaluation and residual analysis.

## Project Overview

The goal of this project is to build a regression model that estimates house sale prices based on property characteristics.

The project focuses on fundamental machine learning concepts, including:
- Exploratory data inspection
- Handling missing values
- Numerical and categorical feature preprocessing
- One-Hot Encoding
- Training and evaluating a Linear Regression model
- Visualizing predictions and analyzing residuals

## Dataset

This project uses the **House Prices: Advanced Regression Techniques** dataset from Kaggle.

- **Source:** [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- **Training samples:** 1,460
- **Original features:** 79 predictors, excluding the ID and target columns
- **Target variable:** `SalePrice`

The target variable represents the final sale price of each house.

### Data Availability

The dataset is not included in this repository. To run the notebook:

1. Download the dataset from Kaggle.
2. Accept the competition rules if required.
3. Place `train.csv` in the `data/` directory.

The expected file structure is:

```text
house-price-linear-regression/
├── data/
│   └── train.csv
├── notebooks/
│   └── 01_house_price_analysis.ipynb
├── src/
├── results/
├── .gitignore
└── README.md
```

## Methodology

### 1. Data Inspection

Inspected the dataset's dimensions, data types, and missing values to understand its structure before preprocessing.

### 2. Data Preprocessing

- Separated the target variable (`SalePrice`) from the input features.
- Removed the `Id` column because it is an identifier rather than a meaningful predictive feature.
- Treated `MSSubClass` as a categorical feature despite its integer data type.
- Split the data into training and testing sets using an 80/20 split with `random_state=42`.

### 3. Missing Value Imputation

- **Numerical features:** Missing values were imputed using the median.
- **Categorical features:** Missing values were filled with the category `"None"` to represent missing or absent property features.

Imputation parameters were learned from the training set and then applied to the test set to avoid data leakage.

### 4. Categorical Encoding

Applied One-Hot Encoding to categorical features using `OneHotEncoder(handle_unknown="ignore")`.

This approach converts categorical values into numerical features without imposing an artificial ordinal relationship between categories.

### 5. Model Training

Trained a baseline `LinearRegression` model using scikit-learn.

### 6. Model Evaluation

Evaluated the model on the held-out test set using:

- **Mean Absolute Error (MAE):** Average absolute difference between actual and predicted prices.
- **Root Mean Squared Error (RMSE):** Penalizes larger prediction errors more heavily.
- **R² Score:** Measures how much of the target's variance is explained by the model relative to a constant-mean baseline.

Residual analysis was also performed to inspect prediction errors and identify potential patterns.

## Results

The baseline model achieved the following results on the held-out test set:

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 20,836.87 |
| Root Mean Squared Error (RMSE) | 30,398.39 |
| R² Score | 0.8676 |

### Interpretation

- The model achieved an R² score of approximately **0.868** on the test set.
- The MAE indicates that the average absolute prediction error was approximately 20,837 price units.
- The RMSE was higher than the MAE, reflecting the greater impact of larger prediction errors.

These results provide a baseline for future experimentation. Performance may vary with different train-test splits, preprocessing choices, and model configurations.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- Jupyter Notebook

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/Sobhankhedry/house-price-linear-regression.git
cd house-price-linear-regression
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter
```

### 4. Download the dataset

Download `train.csv` from the Kaggle competition and place it in the `data/` directory.

### 5. Run the notebook

Launch Jupyter:

```bash
jupyter notebook
```

Open `notebooks/01_house_price_analysis.ipynb` and run the cells in order.

## Limitations

- The current implementation uses a single train-test split.
- Linear Regression may not capture complex nonlinear relationships between property features and sale prices.
- The preprocessing workflow is implemented directly in the notebook rather than consolidated into a reusable scikit-learn pipeline.
- Further validation and feature engineering may improve the model's reliability and predictive performance.

## Future Improvements

Potential next steps include:

- Implementing cross-validation
- Building a reusable preprocessing and modeling pipeline
- Comparing Linear Regression with Ridge and Lasso regression
- Exploring log transformation of the target variable
- Performing more detailed feature and residual analysis
- Adding automated tests and reproducible experiment tracking

## License

The source code in this repository may be used for educational purposes. The dataset remains subject to the terms and conditions of the original Kaggle competition.
