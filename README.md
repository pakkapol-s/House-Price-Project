#  SalaryScope: Predicting Data Science Compensation with Machine Learning

## Goal
To analyze data science salary trends and predict salaries using a machine learning model: Random Forest Regressor, providing insights for job seekers and employers based on role, experience, company size, and working conditions.

---

## Background
The global demand for data professionals continues to surge as industries embrace AI, analytics, and digital transformation. Salary benchmarks play a crucial role in talent acquisition, retention, and strategic career planning.

This project uses a real-world dataset with over 59,000 entries (2020–2025), detailing job titles, experience levels, company size, employment type, and salary data. The aim is to uncover key salary trends and build a machine learning model to predict salaries accurately.

The dataset was thoroughly cleaned, transformed, and engineered. This includes removing outliers, one-hot encoding categorical variables, frequency encoding job titles, changing data types, and creating new features like `is_fully_remote` and `is_domestic`.

---

## Insights Summary

### Salary by Experience
- Experience level has a strong impact on salary: Globally, Experts average $204k, while Entry-level roles earn around $104k. 
- In the US, Mid-level and Senior roles earn slightly more than the global average, indicating a competitive job market.
- UK (GB) stands out: Mid-level roles earn the highest average ($185k), even more than Experts, possibly reflecting market shortages or reporting bias.
- Entry-level salaries are consistently the lowest across all regions, but vary from ~$97k in the UK to ~$106k in the US.

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
- Jupyter / Kaggle Notebook

---

## Author
Mr. Pakkapol Satthapiti
MSC of Data Science and AI | The University of Liverpool | Feel free to connect!