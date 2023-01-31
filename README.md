# gpt_test
- This project is for training apt model and testing chatbot
- The research process can be seen here: https://www.notion.so/toy-project-Chatbot-with-GPT-6fa08edd35c2442ebb2d3fa92e85b4df

## Directory Structure
<img width="1151" alt="image" src="https://user-images.githubusercontent.com/43153661/215898498-003a7a9b-9882-4b48-986a-93fc3b638f67.png">

- [input]
  - movies.csv -> information of movies(title, genre)
  - ratings.csv -> rating data of each user
- [pipelines/chatbot]
  - preprocessing.py -> data preprocessing
  - train.py -> train chatbot (gpt2)
  - deploy_model.py -> deploy chatbot model to sagemaker
  - pipeline.py -> sagemaker pipeline deploy


