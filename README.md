#  SalaryScope: Predicting Data Science Compensation with Machine Learning

## Goal
Through exploratory data analysis and machine learning (Logistic Regression, Random Forest, SVM, XGBoost), this project predicts house sale status. It delivers insights from key property features, benefiting real estate professionals and house seekers

---

## Background
Understanding the dynamics of the residential real estate market is crucial for investors, agents, and prospective homeowners alike. Predicting whether a property will be sold, and what factors influence that outcome, can provide significant strategic advantages in a competitive market.

This project utilizes a real-world dataset covering various USA house listings,specifially in California, detailing essential property features, listing specifics, and crucially, their eventual sale status. The primary objective is to leverage this data to build robust machine learning models capable of predicting property sales outcomes accurately.

A core challenge encountered was addressing critical data integrity issues, specifically geographic inconsistencies where listed cities did not always match their corresponding states. Extensive data preprocessing, meticulous feature engineering (such as deriving price ratios), and comprehensive model tuning were undertaken to overcome these obstacles and enhance the dataset's predictive signal.

---

<!-- ## Insights Summary

### Newer Homes Don’t Always Mean Higher Sale Probability
Contrary to expectations, properties built after 2010 do not always show a higher likelihood of being sold. Sale outcomes appear influenced more by other factors like size, location, and market exposure than age alone.

### Remote Work Impact
- Top-paying roles exist in both fully remote and fully on-site settings, but the highest salary overall ($450,000) is for an on-site position: Research Team Lead.
- Leadership and executive roles dominate both categories, but on-site jobs seem to edge out in salary for traditional management titles like Director of Data and Head of Machine Learning.
- Some roles like Head of AI and Director of Product Management appear in both top 10 lists, showing strong salary consistency regardless of work arrangement.


### Geography Matters
- The U.S. leads in top-tier salaries, with roles like Research Team Lead and Director-level positions earning up to $450,000, noticeably higher than the global and UK equivalents 
- Globally, salary extremes are wider — roles such as Clinical Data Operator and AI Content Writer earn less than $45,000, while executive-level titles exceed $270,000. 
- Job title prestige doesn’t always align across countries: roles like Director and AI Engineer appear in lower-paid UK lists, but rank among top earners in the U.S., highlighting regional valuation differences.

### Job Title Frequency and Pay
- Data Scientist, Data Engineer, and Data Analyst are the most common roles in the dataset, with over 5,000 entries each, showing strong demand across organizations.
- Despite their popularity, these roles aren’t the highest-paid. Data Analyst, in particular, ranks last in average pay among the top 10, earning ~$111K, while Data Scientist and Data Engineer trail behind others at ~$160K–155K.
- Product Manager and Software Engineer balance both decent frequency and high pay, suggesting strong market value for both strategic and technical skill sets.

---
## Recommendations
- **For Job Seekers**: Don't assume popular roles like Data Analyst or Data Engineer guarantee top pay. Instead, explore high-value niches like Machine Learning Engineer or Research Scientist, which show significantly higher average salaries despite being less common.
- **For Employers**: Consider offering fully remote options. Your data suggests that 100% remote roles are often linked with higher salaries, potentially signaling companies’ willingness to invest more in global talent for flexible arrangements.
- **For Global Talent**: Be strategic about geographic positioning. Jobs in the US consistently offer the highest salaries, followed by the UK. If relocation or remote contracting is an option, it could lead to substantial income boosts.
- **For Entry-Level Candidates**: Surprisingly, small companies appear to offer the highest average salary for entry-level positions among all company sizes shown for EN roles. 
—

## Model Results and Summary

The final model was a Random Forest Regressor trained after extensive preprocessing and feature engineering. A log transformation was applied to the salary target variable to improve prediction stability.
- **Mean Absolute Error (MAE)**: ~45,800.06 USD
-**Mean Squared Error (MSE)**: ~3.56 billion
-**R² Score**: ~0.24
While the model doesn’t explain all salary variation, it successfully captures several key patterns. Further improvements could be achieved by incorporating more granular features such as education level, company reputation, or specific job functions.

![sample of grid search code](grid_search.png)

![Model's results](models_result.png)
---

## Data Preprocessing Steps
- **For descriptive analysis and visualizations**
- Converted string types columns like experience_level, employment_type, and job_title
- Dropped unnecessary columns (`salary`, `salary_currency`)
- Converted string types
- Filtered rows for USD only
- Removed duplicates and outliers

- **For the Random Forest Regressor**
- Engineered features: `is_fully_remote`, `is_domestic`
- One-hot encoded `experience_level`, `employment_type`, `company_size`, `employee_residence`, `company_location`
- Frequency encoded `job_title`
- Applied log transformation to `salary_in_usd`

---

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