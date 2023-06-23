#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine,correlation


st.title('Book Recommendation System')
#Importing data
user=pd.read_csv(r"C:\Users\priya\Downloads\Dataset (1)\Users.csv")
book=pd.read_csv(r"C:\Users\priya\Downloads\Dataset (1)\Books.csv")
rating=pd.read_csv(r"C:\Users\priya\Downloads\Dataset (1)\Ratings.csv")


#Applying filter condition for Nearneighbour clustering model
x=rating['User-ID'].value_counts()>200      #Filtering only users who has reviewed minimum of 200 books
y=x[x].index                                      #boolean indexing
ratings=rating.copy()  #creating a copy of rating data set
ratings=ratings[ratings['User-ID'].isin(y)]        #Applying Filter in the dataset

#Merging the data
df1=ratings.merge(book,on='ISBN')


#Applying filter based on the books
no_rating=df1.groupby('Book-Title')['Book-Rating'].count().reset_index()
no_rating.rename(columns={'Book-Rating':'no_of_ratings'},inplace=True)   
#Creating a column which show the total no of rating for that particular book
final_rating=df1.merge(no_rating,on='Book-Title')
#Selecting only is the number of ratings is greater than 50
final_rating=final_rating[final_rating['no_of_ratings']>=50]
#Deleting records of people who have rated the same book multiple times
final_rating.drop_duplicates(['Book-Title','User-ID'],inplace=True)

#Creating pivot table
book_pivot=final_rating.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')
#There are total 742 books and 888 user got filter for suggestion list
book_pivot.fillna(0,inplace=True)


#Creating sparse matrix
book_sparse=csr_matrix(book_pivot)  #To consider only value and to avoid 0 to reduce computation time


#Building the model
model=NearestNeighbors(algorithm='brute')
model.fit(book_sparse)

#Function which takes book name and return suggestion

def recommend_book(book_name):
  suggestion_name=[]
  id=np.where(book_pivot.index==book_name)[0][0]
  distance,suggestion=model.kneighbors(book_pivot.iloc[id,:].values.reshape(1,-1),n_neighbors=6)
  st.subheader("Book's You may like....")
  for i in range(len(suggestion)):
    suggestion_name.extend(book_pivot.index[suggestion[i]])
  return suggestion_name

#recommend_book("Animal Farm")

#Option to choose between item based/ user based recommendation 
mode_of_recommendation=st.selectbox('Select Type of Recommendation',['Book-Based','User-based','Popular-Books','Author Based'])




def display_book(suggested_book):
    #Creating a books data set which does not have any duplicate books
    books=book.drop_duplicates(['Book-Title'])
    name_of_book=[]
    book_cover=[]
    for i in range(len(suggested_book)):
        #Gives a data frame with satisfy that condition
        filter1=books[books['Book-Title']==suggested_book[i]]
        name_of_book.append(filter1['Book-Title'].values[0])
        book_cover.append(filter1['Image-URL-L'].values[0])
    return name_of_book,book_cover


#Display 5 separate columns 
def Book_display(book_name,book_cover):
    col1,col2,col3,col4,col5=st.columns(5)
    with col1:
        st.image(book_cover[1],use_column_width=True,caption=book_name[1])
    with col2:
        st.image(book_cover[2],use_column_width=True,caption=book_name[2])
    with col3:
        st.image(book_cover[3],use_column_width=True,caption=book_name[3])
    with col4:
        st.image(book_cover[4],use_column_width=True,caption=book_name[4])
    with col5:
        st.image(book_cover[5],use_column_width=True,caption=book_name[5])

#$$$$$$$$---------------------------------------------------------------$$$$$$$$$#

#Display 5 separate columns 
def Book_display_popular(book_name,book_cover):
    col1,col2,col3,col4,col5=st.columns(5)
    with col1:
        st.image(book_cover[0],use_column_width=True,caption=book_name[0])
    with col2:
        st.image(book_cover[1],use_column_width=True,caption=book_name[1])
    with col3:
        st.image(book_cover[2],use_column_width=True,caption=book_name[2])
    with col4:
        st.image(book_cover[3],use_column_width=True,caption=book_name[3])
    with col5:
        st.image(book_cover[4],use_column_width=True,caption=book_name[4])






# User based recommendation definition

user_based_pivot=final_rating.pivot_table(index='User-ID',columns='Book-Title',values='Book-Rating')
user_based_pivot.fillna(0,inplace=True)
#Give the distance between users, min distance indicate that user's have similar taste
user_similarity=1-pairwise_distances(user_based_pivot.values,metric='cosine')
user_similar_df=pd.DataFrame(user_similarity)
user_id_list=final_rating['User-ID'].unique()    #To get all user id
#Setting the index and columns to user id
user_similar_df.index=final_rating['User-ID'].unique()
user_similar_df.columns=final_rating['User-ID'].unique()
#Filling the diagonal value with 0 
np.fill_diagonal(user_similarity,0)
def user_based_recommender(user_id):
    #Input from customer and finding the similar member id
    user1=user_similar_df[user_similar_df.index==user_id]
    temp=user1.idxmax(axis=1)
    k1=int(temp)
#creating data frame for the above 2 id(input and similar id)
    user_in=final_rating[final_rating['User-ID']==user_id]
    user_sim=final_rating[final_rating['User-ID']==k1]
#Merging the 2 data frame on outer join
    suggest_df=pd.merge(user_sim,user_in,on='Book-Title',how='outer')
    #Selecting only book with min 5 rating
    temp_df=suggest_df[(suggest_df['Book-Rating_x']>5)]    #data frame to store suggestion table
    #Filling all the nan values with null,inorder to apply filter in the next stage
    temp_df=temp_df.fillna('Null')
    #Filtering so that only book which are not read by the user will be suggested
    temp_df=temp_df[temp_df['User-ID_y']=='Null']
#Sorting based on decreasing order of rating made by similar user and taking the first 5 book if exist
    book_list=temp_df.sort_values('Book-Rating_x',ascending=False).head(5)
    book_list=book_list['Book-Title'].values
    book_list=list(book_list)
    st.subheader("Book's You may like....")
    return book_list

#$$$$$$$$---------------------------------------------------------------$$$$$$$$$#


#&&&&&&&&&&&&&--------------------------weighted average method --------------&&&&&&&&&&&#####
#Creating a separate data for weighted average
popular_df=final_rating.copy()
popular_df['average_rating']=popular_df.groupby('Book-Title')['Book-Rating'].transform('mean')   #Creating r for the equation
popular_df['no_of_ratings']=popular_df.groupby('Book-Title')['Book-Rating'].transform('count')   #Creating v for the equation
v=popular_df['no_of_ratings']
R=popular_df['average_rating']
c=popular_df['average_rating'].mean()
m=200    #Set min number of votes
#Creating a table which hold's a weighted average 
popular_df['weighted_average']=((R*v)+(c*m))/(v+m)
#Removing all the duplicate book title from the data
popular_books=popular_df.sort_values('weighted_average',ascending=False).drop_duplicates('Book-Title').reset_index()
#These are the top 20 recommendation based on weighted average
unique_authors=popular_books['Book-Author'].unique()            #Getting all the author for author based recommender system
global_popular_books=popular_books['Book-Title'].values
global_popular_books=list(global_popular_books)

#&&&&&&&&&&&&&--------------------------weighted average method --------------&&&&&&&&&&&#####

####-----Author based recommender system


rating=pd.read_csv(r"C:\Users\priya\Downloads\Dataset (1)\Ratings.csv")
#Merging the data
df_author=user.merge(rating,on='User-ID')
df_author=df_author.merge(book,on='ISBN')
df_author['Average_rating']=df_author.groupby('Book-Title')['Book-Rating'].transform('mean')
df_author['Number_of_rating']=df_author.groupby('Book-Title')['Book-Rating'].transform('count')
#Minimum number of vote
v=df_author['Number_of_rating']
R=df_author['Average_rating']
c=df_author['Average_rating'].mean()
m=200    #Set min number of votes
df_author['weighted_average']=((R*v)+(c*m))/(v+m)
df_author.drop(['Age','Location','Image-URL-S','Image-URL-M'],axis=1,inplace=True)
df_author.drop_duplicates('Book-Title',inplace=True)
df_author=df_author.sort_values('weighted_average',ascending=False)


def author_based_recommender(author_name):
    
    selected_df=df_author[df_author['Book-Author']==author_name]
    selected_df=selected_df.drop_duplicates('Book-Title')
    book_list=selected_df['Book-Title'].values
    book_list=list(book_list)
    return book_list


#implimenting for loop for function calling
if mode_of_recommendation=='Book-Based':
    book_list=book_pivot.index.values                #List of book available to search and getting input from user
    book_rec=st.selectbox('Book Name',book_list)     #Allows user to select book that he has read
    suggested_books=recommend_book(book_rec)          #selected book is given to the function which returns suggested books
    book_name,book_cover=display_book(suggested_books) #Book name,cover photo of the book will be extracted
    Book_display(book_name,book_cover)                 #Display the book name cover photo
elif mode_of_recommendation=='User-based':
    user_id=st.selectbox('Enter user ID',user_id_list)
    user_based_book_list=[]
    try:                    #exception handiling adds books from popular books in case the similar guy can refer book less than 5 nos
        user_based_book_list=user_based_recommender(user_id)
    except:
        user_based_book_list=[]
    user_based_book_list.extend(global_popular_books)
    user_based_book_list=user_based_book_list[:6]
    book_name,book_cover=display_book(user_based_book_list) #Book name,cover photo of the book will be extracted
    Book_display(book_name,book_cover)
elif mode_of_recommendation=='Author Based':
    selected_author=st.selectbox('Select the author',unique_authors)
    suggested_books=author_based_recommender(selected_author)
    book_name,book_cover=display_book(suggested_books)
    Book_display_popular(book_name[:6],book_cover[:6])
else:
     book_name,book_cover=display_book(global_popular_books)
     st.subheader('Popular Books')
     for i in range(0,50,5):
        Book_display_popular(book_name[i:i+5],book_cover[i:i+5])
 


# In[ ]:




