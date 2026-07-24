# Credit Card Customer Segmentation — Week 4 Task

## Overview
This project applies unsupervised learning to segment ~9,000 credit card holders into meaningful behavioral groups using K-Means and Hierarchical Clustering, based on 6 months of spending and payment data. Unlike previous weeks, there is no target variable — the goal is to discover natural patterns in customer behavior for use in marketing, risk management, and personalization.

## Dataset
- **Name:** Credit Card Dataset for Clustering
- **Source:** Kaggle — https://www.kaggle.com/datasets/arjunbhasin2013/ccdata
- **Records:** 8,950 active credit card holders, 18 behavioral features

## Part 1 — K-Means Clustering
- Dropped `CUST_ID`, handled missing values in `CREDIT_LIMIT` and `MINIMUM_PAYMENTS` using median imputation
- Scaled all features using `StandardScaler` (mandatory for distance-based clustering)
- Ran K-Means for k=2 to 10, plotted the Elbow Curve and Silhouette Scores
- Selected **k=4** as the optimal number of clusters, balancing the elbow bend and interpretability
- Profiled each cluster with a heatmap of mean feature values

### Cluster Summary
| Cluster | Segment | Description |
|---|---|---|
| 0 | Low-Activity / Inactive | Modest balance, very low purchase activity |
| 1 | High Spenders / Premium | Highest purchases, credit limit, and payments |
| 2 | Cash Advance Users (High Risk) | Highest cash advance usage, never pays in full |
| 3 | Frequent Installment Shoppers | Low balance, high installment purchase frequency |

## Part 2 — Hierarchical Clustering
- Applied Agglomerative Hierarchical Clustering (Ward's method) on a random sample of 300 rows
- Visualized results with a dendrogram, with a threshold line marking the cut for 4 clusters
- Compared against K-Means labels using a cross-tabulation

**Finding:** K-Means and Hierarchical Clustering showed moderate-to-strong agreement for 3 of the 4 clusters, but Hierarchical Clustering produced one very small (outlier) cluster and split one K-Means cluster across two groups. K-Means was recommended for production use due to its scalability and balanced cluster sizes.

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebooks/week4_clustering.ipynb
```

## Author
Musfira Malik — AI/ML Internship
