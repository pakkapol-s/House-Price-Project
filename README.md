# "Built to Sell: ML for House Status

## Goal
Through exploratory data analysis and machine learning (Logistic Regression, Random Forest, SVM, XGBoost), this project predicts house sale status. It delivers insights from key property features, benefiting real estate professionals and house seekers

---

## Background
Understanding the dynamics of the residential real estate market is crucial for investors, agents, and prospective homeowners alike. Predicting whether a property will be sold, and what factors influence that outcome, can provide significant strategic advantages in a competitive market.

This project utilizes a real-world dataset covering various USA house listings,specifially in California, detailing essential property features, listing specifics, and crucially, their eventual sale status. The primary objective is to leverage this data to build robust machine learning models capable of predicting property sales outcomes accurately.

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

- Column Renaming: Long column names were standardized (e.g., "Area (Sqft)" → "area_sqft", "Lot Size" → "lot_size").
- Data Type Conversion: Key fields like "City", "State", "Property Type" were explicitly cast as strings for clarity and consistency.
- Numeric Extraction: Bedroom, bathroom, area, and lot size values were cleaned by stripping non-numeric characters (e.g., “3 Beds” → 3).
- Currency Conversion: The "Price" column was stripped of $ and , symbols and converted to numeric values (float).
- Address Breakdown: Extracted the street component from the full address for added granularity in location analysis.
- Geographic Correction: Fixed mismatched city–state pairs (e.g., San Diego mistakenly tagged as TX instead of CA).
- Dropped Redundant Columns: Removed unused original columns after transformation to keep the dataset lean and tidy.

#### 2. Data Preprocessing for Binary Classification Models

Building upon the general cleaning, additional steps were performed to prepare the data specifically for training machine learning models to predict `is_sold` status:
    
* **Category Encoding**:
    * City, Property_type, and Status converted to categorical data types.
    * One-Hot Encoding (OHE) applied to selected categorical features, with drop='first' to avoid multicollinearity.
* **Feature Engineering**:
    * Created log_price using a log1p transformation for better numeric stability.
    * Derived new binary target variable is_sold from the original "Status" field.
* **Column Reduction**:
    * Removed unnecessary columns for modeling (e.g., URLs, Agent Names, raw Price) to avoid data leakage and reduce noise.
* **Data Cleaning**:
    * Applied consistent formatting across numerical features.
    * Handled geographic inconsistencies similarly to the EDA version.
* **Column Reordering**:
    * Reordered columns to ensure is_sold is at the end of the DataFrame, improving readability during training.

---

<!-- 
## Visualizations
- Remark: visualisations are based on data in the year of 2025
1. Distribution of company size
![Distribution of company size plot](company_dis.png)
2. Salary distribution
![Salary distribution plot](salary_distribution.png)
3. Salary by experience level
![Salary by experience level plot](salary_by_experience_level.png)
4. Average Salary by country 
![Average Salary by country](average_salary_by_country.png)
5. Heatmap: Salary VS Experience VS Company Size
![Heatmap: Salary VS Experience VS Company Size](heatmap.png)
6. Bar Plot of Top 10 Highest paid job titles
![Top 10 highest job title](top_jobtitiles.png)
7. Remote Work & Salary
![Remote work & Salary plot](remote_work_salary.png)
8. Salary Over Time by Experience Level
![Salary over time plot](salary_trends.png)

---

## Challenges Faced
- High cardinality in `job_title` required frequency encoding
- Skewed salary distribution handled using log transformation
- Outlier handling dropped ~2,900 rows
- Low R² score indicates the dataset has high variance not captured by current features
- missing information in some countries like "CA" - contains only one row, "NL" - contains only 2 rows.
![Sample rows of some countries that contain only few rows](problem.png)

---

## Technologies Used
- Python, Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter / Kaggle Notebook -->

---

## Author
Mr. Pakkapol Satthapiti
MSC of Data Science and AI | The University of Liverpool | Feel free to connect!