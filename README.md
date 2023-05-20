# Recommendation_system
A simple content based recommendation system made for IPD project Library management system

## The recommendation system does not use any machine learning but simple NLP(Natural Language Processing). The dataset was first cleaned and all null values and everything were removed.
## Due to the lack of data, We combined the author,Book Name and Publisher and the Domain of the books to make a new column which was then used to make the recommendation system.
## After this, We converted the new column to a vector using the sentence-transformers package(BERT embeddings) and then used the cosine similarity function to find the cosine similarity matrix.
## Then use that matrix to find the index of the books which are closest to the book given by the user.

### The pickle is there to store the cosine similarity matrix after running the code so as to not make the matrix everytime to get a recommendation.

