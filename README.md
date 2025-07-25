# "Built to Sell: ML for House Status

## Goal
Through exploratory data analysis and machine learning (Logistic Regression, Random Forest, SVM, XGBoost), this project predicts property sale status. It delivers insights from key property features, benefiting real estate professionals and house seekers

---

## Background
Understanding the dynamics of the residential real estate market is crucial for investors, agents, and prospective homeowners alike. Predicting whether a property will be sold, and what factors influence that outcome, can provide significant strategic advantages in a competitive market.

This project utilizes a real-world dataset covering various USA house listings, specifically in California, detailing essential property features, listing specifics, and crucially, their eventual sale status. The primary objective is to leverage this data to build robust machine learning models capable of predicting property sales outcomes accurately.

A core challenge encountered was addressing critical data integrity issues, specifically geographic inconsistencies where listed cities did not always match their corresponding states. Extensive data preprocessing, meticulous feature engineering (such as deriving price ratios), and comprehensive model tuning were undertaken to overcome these obstacles and enhance the dataset's predictive signal.

---

## Insights Summary

### Location Influence
- Sacramento and Los Angeles had the highest share of sold listings, suggesting higher demand in these markets compared to cities like Fresno or San Francisco.
- San Francisco underperforms in average price even though it is often considered a premium market, its average sale price ($788k) is the lowest among the five cities. That may reflect a concentration of lower-value properties sold recently or a cooling trend in that segment of the market.
- Sacramento is a more liquid market — homes sell more frequently but at slightly lower price points, its average price ($804k) is slightly lower than Fresno, LA, and San Diego.

### Property Size and Configuration
- While condos have the lowest average total price, they don’t offer the best value per square foot. Single-family homes have the highest price per sqft, showing they command premium space.
- Apartments in Fresno ($912K) and single-family homes in San Diego ($865K) rank among the most expensive. On the flip side, condos in Sacramento and San Francisco consistently appear in the lowest price brackets.
- 2-bedroom homes have the highest median sale price (~$870K), even more than 4- or 6-bedroom properties. This suggests smaller, well-located or luxury properties are in high demand over sheer size.

### Property Age and Market Trends
- A significant 67% of homes were built before 2000 (2,013 out of 3,000), showing the market leans toward established, possibly renovated properties rather than new developments.
- Old doesn’t mean cheap — older homes average higher prices ($814K) than newer ones ($803K), suggesting buyers may be paying for land, location, or charm over modern construction.
- The price difference between old and new homes is most pronounced in single-family homes, where older units cost ~$25K more. This may reflect larger lot sizes or better neighborhoods.

### Underprice Segment Analysis
- Apartments in San Francisco and San Diego are significantly underpriced, averaging $570K–$633K, far below cities like Fresno. Meanwhile, townhouses in Los Angeles average only $527K, the lowest across all segments, revealing strong bargain potential.
- With average prices ranging from $557K in Sacramento to $660K in Fresno, condos emerge as one of the most underpriced and accessible property types, especially for entry-level buyers or investors targeting rental yields.
- Despite being a high-cost city, San Francisco properties (apartments, condos, and single-family homes) appear in the lowest average underpriced listings, indicating market-wide price corrections or opportunity pockets even in premium neighborhoods.

---

## Recommendations
- For investors and first-time buyers, the under-price analysis highlights promising opportunities within the apartment and townhouse segments of Los Angeles and San Francisco. Highly recommend exploring these areas for potential appreciation and value gains.
- For entry-level investors or buyers: Condos in Sacramento and Fresno show consistently lower average prices per square foot, making them ideal for seeking affordability with steady long-term potential.

- Locations matter: While the number of bedrooms showed some correlation with price, city-level influence was far more significant. Prioritize location over layout when filtering potential properties—especially in fluctuating markets like Los Angeles and San Francisco.
- House old matters: Contrary to common belief, newer properties (built ≥ 2000) are not always more expensive. In fact, older homes averaged slightly higher prices. Buyers should assess property condition and value, not just age, when making decisions.

---

## Data Preprocessing Steps

The raw dataset underwent comprehensive preprocessing to ensure data quality, consistency, and suitability for both exploratory data analysis and machine learning model training. These steps were divided based on the analytical objective:

#### 1. Preprocessing for General Descriptive Analysis (EDA)

The goal here was to clean and transform raw property listing data for EDA and insight generation, the following transformations were applied:

- Column Renaming: Long column names were standardized (e.g., `Area (Sqft)` → `area_sqft`, `Lot Size` → `lot_size`).
- Data Type Conversion: Key fields like `City`, `State`, `Property Type` were explicitly cast as strings for clarity and consistency.
- Numeric Extraction: Bedroom, bathroom, area, and lot size values were cleaned by stripping non-numeric characters (e.g., “3 Beds” → 3).
- Currency Conversion: The `Price` column was stripped of $ and , symbols and converted to numeric values (float), and renamed to `price_in_usd`.
- Address Breakdown: Extracted the street component from the full address for added granularity in location analysis.
- Geographic Correction: Fixed mismatched city–state pairs (e.g., San Diego mistakenly tagged as TX instead of CA).
- Dropped Redundant Columns: Removed unused original columns after transformation to keep the dataset lean and tidy.

#### 2. Data Preprocessing for Binary Classification Models

Building upon the general cleaning, additional steps were performed to prepare the data specifically for training machine learning models to predict `is_sold` status:
    
* **Category Encoding**:
    * Columns like `City`, `Property_type`, and `Status` converted to categorical data types.
    * One-Hot Encoding (OHE) applied to selected categorical features, with drop='first' to avoid multicollinearity.
* **Feature Engineering**:
    * Created log_price using a log1p transformation for better numeric stability.
    * Derived new binary target variable is_sold from the original "Status" field.
* **Column Reduction**:
    * Removed unnecessary columns for modeling (e.g., `URLs`, `Agent Names`, raw `Price`) to avoid data leakage and reduce noise.
* **Data Cleaning**:
    * Applied consistent formatting across numerical features.
    * Handled geographic inconsistencies similarly to the EDA version.
* **Column Reordering**:
    * Reordered columns to ensure is_sold is at the end of the DataFrame, improving readability during training.

---

## Machine Learning Model Evaluation

This project utilized several classification algorithms to predict the `is_sold` status of properties. Each model was fine-tuned using GridSearchCV and evaluated on a hold-out test set to assess its generalization capabilities.

### Performance Summary

| Model                   | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|-------------------------|----------|-----------|--------|----------|---------|
| Logistic Regression     | 0.4958   | 0.3508    | 0.5169 | 0.4180   | 0.5026  |
| Random Forest Classifier| 0.6024   | 0.3772    | 0.2077 | 0.2679   | 0.5135  |
| Support Vector Machine  | 0.3486   | 0.3455    | 0.9614 | 0.5083   | 0.5210  |
| XGBoost Classifier      | 0.59     | 0.34 (for class 1) | 0.17 (for class 1) | 0.23 (for class 1) | 0.49 (macro avg AUC) |

**Key Findings & Discussion:**

As noted in the "Challenges Faced" section, a significant observation was the **limited predictive power** of all models in classifying property sale status.
* All models exhibited F1-Scores generally below 0.5 and ROC AUC scores only marginally better than random guessing (~0.50-0.52).
* The SVM model achieved the highest F1-Score (0.5083), but its very high recall (0.9614) paired with low precision (0.3455) suggests it likely classified a large majority of instances into the positive class, indicating a potential imbalance or difficulty in distinguishing true positives.
* Random Forest and XGBoost, while often powerful, also struggled, particularly with the positive class (F1-scores around 0.23-0.27), indicating a low ability to correctly identify properties that were actually sold.

These results strongly suggest that the current dataset, even after extensive preprocessing and feature engineering, may lack sufficiently strong or diverse signals to accurately predict the binary `is_sold` outcome. Further improvements would likely necessitate additional, more granular data sources (e.g., market sentiment indicators, interest rates, property age, detailed neighborhood demographics, agent marketing efforts, etc.) to capture the complex dynamics of property sales.

---

## Visualizations
- Remark: visualisations are based on data in the year of 2025
1. Property Type Price Distributiuon
![Property Type Price Distributiuon plot](price_distribution.png)
2. Average Price per Sqft by Year Built
![Average Price per Sqft by Year Built](Average_Price_per_Sqft_by_Year_Built.png)
3. Price Distribution By Property Type
![Price Distribution By Property Type plot](Price_Distributiom_By_Property_Type.png) 
4. Property Type VS Average Price (USD)
![Property Type VS Average Price (USD) plot](Property_Type_VS_Average_Price_(USD).png)
5. Average Price by The Number of Bedrooms
![Average Price by The Number of Bedrooms](Number_of_Bedrooms_vs._Average_Price_(USD).png)
6. Old vs New House Average Price Comparison
![Old vs New House Average Price Comparison Plot](Old_vs_New_House_Price_Comparison.png)
7. City VS. Average Price
![City VS. Average Price plot](City_VS._Average_Price_Plot.png)
8. Underpriced properties in each city:
![Underpriced properties in each city plot](underpriced_prop_in_each_cities.png)
9. Correlation Heatmap For House Price
![Correlation Heatmap For House Price](correlation_heatmap.png)
10. Property Type Distribution Over the Years
![Property Type Distribution Over the Years Plot](Property_distribution_over_the_years.png)
11. House Price Trends by Property Type (1950–2023)
![House Price Trends by Property Type (1950–2023) Plot](House_price_trend_plot.png)

---

## Challenges Faced

### 1. Data Integrity

The dataset includes City and State columns. While there are 5 unique City values (Fresno, Los Angeles, Sacramento, San Diego, San Francisco) and 5 unique State values (CA, FL, IL, NY, TX), all the listed cities are geographically located within California (CA). For example, I've found instances where 'Los Angeles' is incorrectly listed with 'NY' as its state. This is a factual inconsistency that impacts the reliability of the data.

In order to solve this problem, I reassigned State to 'CA' for all entries. Since all the City values in the dataset are definitively in California, I can programmatically reassign the State column to 'CA' for every row, overriding the incorrect state values. This ensures factual accuracy and consistency across the dataset.

Sample of mismatched rows
![mismatched rows](incorrect_state_map.png)

### 2. Defining “Underpriced” Properties

Since there is no universal definition of what makes a property “underpriced.” I needed to carefully design a logic that flags underpriced listings based on expected price predictions from the model rather than hardcoded rules — requiring thoughtful calibration.

Solution: I did some research and concluded that I would identify underpriced properties based on price per square foot compared to the average for their property type.

### 3. Designing Actionable Insights

Without feedback or business goals, it can be difficult to decide what questions to answer or which analysis to prioritize. Creating meaningful insights required stepping into the role of a real estate analyst. 

Solution: To simulate real-world needs, I stepped into the role of a real estate analyst and asked questions a property investor or buyer might care about, such as:

Which cities have the most underpriced properties?
Which property types tend to be undervalued?
Are newly built homes priced differently than older ones?

### 4. Model's poor performace

Despite the efforts in feature engineering, hyperparameter tuning, and using a powerful model like Random Forest, the performance remains very poor for predicting whether a house will be sold. The low F1-score, low recall for the positive class, and AUC near 0.5 are consistent across your models.

This strongly reinforces the idea that the current dataset, even with the engineered features, likely lacks sufficiently strong predictive signals to accurately determine if a house will be sold.

---

## Technologies Used

- Programming Language: Python
- Data Manipulation & Analysis: Pandas, NumPy
- Data Visualization: Matplotlib, Seaborn
- Statistical Modeling & Preprocessing: Statsmodels (for VIF), Scikit-learn (for preprocessing, model selection, various models, and evaluation metrics)
- Machine Learning Models:
    **Logistic Regression**
    **Random Forest Classifier**
    **Support Vector Classifier (SVC)**
    **XGBoost Classifier**
- Jupyter / Kaggle Notebook
- Dashboard: Tableau

---

## Dashboard: California Underpriced Property Insights

To provide interactive insights into the California housing market, a Tableau Public dashboard was developed, focusing on identifying and analyzing underpriced properties across five key California cities: Fresno, Los Angeles, Sacramento, San Diego, and San Francisco.

The dashboard features the following interactive components:

1. Key Performance Indicators (KPIs) per City: A detailed table presents essential metrics for each of the five cities. This includes the most common property type, the total number of properties, the count of identified underpriced properties, and the proportion of underpriced properties relative to the total. This allows for a quick comparative overview of value opportunities across different urban centers.

2. Geographic Distribution Map: An intuitive map of California visually highlights the five analyzed cities. Users can interact with the map to filter data, making it easy to focus on specific regions of interest. The color intensity on the map can further indicate metrics like the number of underpriced properties in a city.

3. Average Price Comparison Bar Plot: A side-by-side bar chart offers a clear visual comparison between the average listing price and the average price of underpriced properties within each city. This directly quantifies the potential savings or value found in the "underpriced" segment.

4. Underpriced Property Type Heatmap: A detailed heatmap provides a granular view of the average price of underpriced properties, broken down by property type within each city. The color intensity of the cells reveals a higher number of underpriced properties, allowing users to identify specific property types that offer the most value in a given city. Hovering over the heatmap provides tooltips with detailed information like the underprice count, city name, property type, and average price of underpriced property.

- Dashboard link: https://public.tableau.com/views/CaliforniaHousePrice_17528675775100/CalDash?:language=en-GB&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link

1. Sample of the dashboard
![Sample of the dashboard](dash_2.png)
2. Sample Dashboard with interactive elements
![Sample of the dashboard](dash_1.png)

---

## Author
Mr. Pakkapol Satthapiti
MSC of Data Science and AI | The University of Liverpool | Feel free to connect!