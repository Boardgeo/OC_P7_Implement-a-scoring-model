# Implement a scoring model

## Context
The financial company *Prêt à dépenser* offers consumer credit for people with little or no loan history.

The company is interested in identifying customers who will not pay the loan back to minimize the company’s financial losses. Hence, they want to implement a "credit scoring" tool to calculate the probability that a customer will repay his credit, and then classify the request as granted or denied credit.

## Goal:
  - Develop a scoring model to predict the risk of non-repayment of a client.

  - Build an interactive dashboard to interpret the results so that customer relations managers could explain credit decisions as transparently as possible, but also allow their customers to have access to their personal information and explore it easily.

## Objectives:
  - Build a scoring model that will automatically predict the probability of a customer's bankruptcy.

  - Build an interactive dashboard for customer relationship managers to interpret the predictions made by the model, and to improve the customer knowledge of customer relationship managers.

  - Put into production the prediction scoring model using an API, as well as the interactive dashboard that calls the API for predictions.

## Tools and Skills
 - End to end MLOps
 - Experimental tracking - Mlflow
 - Model classification and threshold tunings
 - Custom score/custom metrics
 - Model interpretation - SHAPley
 - Data drift - Evidently
 - FastAPI
 - Streamlit for dashboard 
 - Model deployments - heroku (https://pret-dash-app.herokuapp.com/)
