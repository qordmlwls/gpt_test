# gpt_test
- This project is for training apt model and testing chatbot
- The research process can be seen here: https://www.notion.so/toy-project-Chatbot-with-GPT-6fa08edd35c2442ebb2d3fa92e85b4df

## Directory Structure
<img width="942" alt="Screenshot 2022-12-30 at 2 09 11 PM" src="https://user-images.githubusercontent.com/43153661/210163319-2983750e-0fcf-4eb7-bf9a-c3b85b76e0b0.png">

- [input]
  - movies.csv -> information of movies(title, genre)
  - ratings.csv -> rating data of each user
- [pipelines/chatbot]
  - preprocessing.py -> data preprocessing
  - train.py -> train chatbot (gpt2)
  - deploy_model.py -> deploy chatbot model to sagemaker
  - pipeline.py -> sagemaker pipeline deploy


