# Fake-Job-Postings-Detector
The project aims to build a ML pipeline exposed as an API which can be called to detect fake job postings.


## Dataset used:
[Real / Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)

#### Add manually the airflow user admin
```bash
airflow users create \
  --username admin \
  --password admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com
  ```