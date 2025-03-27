# Sentiment-NLP1
We want to build sentiment analysis services for our corporate customers. 
With the services, Our innovation AI’s partner company will have a better understanding of the users’ perspective on their products and services.
Our potential corporate companies may start with tech startups and extend to larger corporations such as Target, Walmart, Johnson & Johnson, etc. 
They will be able to better optimize the user targeting strategy, budget allocation, hence lifting the companies revenue and brand. 
## Access database through Google Drive
raw: https://drive.google.com/file/d/1VpUyS1HrnjpwEh5MoXHgR_5wKBCUPe1J/view 

cleaned:https://drive.google.com/file/d/1SDlOvsZy715KSQ0zVMYPJu4y6yiPUMwr/view?usp=sharing

## Data Cleaning & visualization 
- Missing values
- Outliers
- Oriented brand name by classifying rare brand (reviews < 300, can be changed in variable `rare_threshold` later) as `others`. Pick top 13 brands as known brands and normalized brand names by using function `unify_brand_name` (mainly by `process` in package `rapidfuzz`).
- Normalized product name by applying regularization and  `process` in package `rapidfuzz`.
- 
