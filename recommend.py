#importing necessary libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sentence_transformers import SentenceTransformer
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA




#putting books data on 'books' dataframe
books = pd.read_csv('new_cleaned_books_data.csv')
# print(len(books))
books_without_nan = books.dropna()
books_without_duplicates = books_without_nan.drop_duplicates(subset="BookName",keep="first")
# print(len(books_without_nan))
# print(books_without_duplicates)




books_without_duplicates["new_column"] =  books_without_duplicates["BookName"] +" "+ books_without_duplicates["BookAuthor"] + " " +  books_without_duplicates["Publication"] 
# print(books_without_duplicates)


# book_data = np.array(books_without_duplicates["new_column"])
# model = SentenceTransformer('distilbert-base-nli-mean-tokens')
# embeddings = model.encode(book_data, show_progress_bar=True)

# cos_sim_data = pd.DataFrame(cosine_similarity(embeddings))
# print(cos_sim_data.shape)
# with open("data.pickle",'wb') as f:
#     pickle.dump(cos_sim_data,f)

def give_recommendations(name,print_recommendation = False,print_recommendation_plots= False,print_genres =False):
  print("VERSION IS ",pd.__version__)
  index = books_without_duplicates.loc[books_without_duplicates["BookName"]==name].index.values[0]
  # print(index)
  with open('data.pickle', 'rb') as f:
     cos_sim_data = pickle.load(f)
  index_recomm =cos_sim_data.loc[index].sort_values(ascending=False).index.tolist()[1:6]
  # print(index_recomm)
  books_recommend =  books_without_duplicates.iloc[index_recomm].values
  books = books_recommend.tolist()
  result = {'Books':books,'Index':index_recomm}
  if print_recommendation==True:
    # print('The watched book is this one: %s \n'%(books_without_duplicates['BookName'].loc[index]))
    k=1
    for book in books_recommend:
      print('The number %i recommended book is this one: %s \n'%(k,book))
  if print_recommendation_plots==True:
    # print('The plot of the watched movie is this one:\n %s \n'%(books_without_duplicates.iloc[index]))
    k=1
    for q in range(len(books_recommend)):
      plot_q = books_without_duplicates.loc[index_recomm[q]]
      print('The plot of the number %i recommended movie is this one:\n %s \n'%(k,plot_q))
      k=k+1
  # if print_genres==True:
  #   print('The genres of the watched movie is this one:\n %s \n'%(data['Genre'].loc[index]))
  #   k=1
  #   for q in range(len(movies_recomm)):
  #     plot_q = data['Genre'].loc[index_recomm[q]]
  #     print('The plot of the number %i recommended movie is this one:\n %s \n'%(k,plot_q))
  #     k=k+1
  return result


give_recommendations("Computer Algorithms",True)



