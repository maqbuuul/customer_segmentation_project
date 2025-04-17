# Customer Segmentation Project

## Overview
This project implements end-to-end customer segmentation using RFM (Recency, Frequency, Monetary) analysis on retail transaction data. The system preprocesses the data, engineers features, applies clustering algorithms, and presents the results through an interactive Streamlit dashboard with actionable business insights.

## Features
- **Data preprocessing pipeline**: Cleans and transforms retail transaction data
- **RFM feature engineering**: Creates and normalizes customer features based on purchase behavior
- **Advanced customer segmentation**: Uses K-means clustering with silhouette analysis
- **Interactive Streamlit dashboard**: Visualizes segments with 3D plots, radar charts, and more
- **Business insights and recommendations**: Provides actionable strategies for each customer segment
- **ROI calculator**: Estimates potential returns from segment-targeted campaigns

## Project Structure
```
customer_segmentation_project/
│
├── data/                         # Data storage
│   ├── raw/                      # Original data files
│   │   └── online_retail_combined_dataset.csv
│   └── processed/                # Cleaned and processed data
│       ├── cleaned_retail_data.csv
│       ├── rfm_data.csv
│       ├── customer_features.csv
│       └── customer_segments.csv
│
├── notebooks/                    # Jupyter notebooks
│   └── exploratory_analysis.ipynb
│
├── src/                          # Source code
│   ├── data/
│   │   ├── data_loader.py        # Functions to load and validate data
│   │   └── data_processor.py     # Data cleaning and preprocessing
│   │
│   ├── features/
│   │   └── feature_engineering.py # RFM features creation
│   │
│   ├── models/
│   │   └── clustering.py         # KMeans and other clustering methods
│   │
│   └── visualization/
│       └── visualize.py          # Functions for visualization
│
├── streamlit/                    # Streamlit dashboard
│   ├── app.py                    # Main dashboard application
│   └── pages/
│       ├── 01_Data_Overview.py   # Data exploration
│       ├── 02_Customer_Segments.py # Segment analysis
│       └── 03_Business_Insights.py # Strategic recommendations
│
├── models/                       # Saved models and scalers
│   ├── kmeans_model.pkl
│   └── rfm_scaler.pkl
│
├── reports/                      # Generated analysis reports
│   └── figures/                  # Generated graphics and plots
│
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/customer-segmentation-project.git
cd customer-segmentation-project
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

4. Place the raw data file in the `data/raw/` directory.

## Usage

### Data Processing
```bash
# Run the data preprocessing pipeline
python -m src.data.data_processor

# Run the feature engineering pipeline
python -m src.features.feature_engineering

# Run the clustering pipeline
python -m src.models.clustering
```

### Running the Dashboard
```bash
streamlit run streamlit/app.py
```

## Dashboard Features

### Main Dashboard
- Key metrics overview (total customers, average values)
- Customer segment distribution
- 3D cluster visualization
- RFM metrics distribution

### Data Overview
- Data quality assessment
- Time series analysis of sales
- Geographic distribution of customers
- Product analysis and trends

### Customer Segments
- Detailed segment profiles
- Interactive comparisons between segments
- Segment characteristic visualization
- Customer explorer for individual analysis

### Business Insights
- Revenue opportunity identification
- Segment-specific strategic recommendations
- Campaign planning and calendar
- ROI calculator for marketing initiatives

## Customer Segments

The project identifies distinct customer segments:

| Segment | Description | Recommended Actions |
|---------|-------------|---------------------|
| Champions | High-value, frequent, recent customers | Reward loyalty, encourage ambassadorship |
| Loyal Customers | Regular, active customers | Increase purchase value, special offers |
| Big Spenders | High-value but infrequent | Increase purchase frequency, personalized communication |
| New Customers | Recent first-time buyers | Convert to repeat customers, second purchase incentives |
| At Risk | Previously active, becoming inactive | Reactivation campaigns, special offers |
| Dormant | Long-inactive customers | Re-engagement, win-back promotions |
| Rare Shoppers | Infrequent, low-value | Increase purchase frequency and value |

## Model Performance
- The clustering model uses silhouette score and inertia to determine optimal number of clusters
- Customer segment labels are validated through business rules and domain knowledge
- Segment recommendations are based on RFM metrics and retail industry best practices

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- The dataset used is the Online Retail dataset from the UCI Machine Learning Repository
- Thanks to all contributors and the open-source community for their valuable tools and libraries