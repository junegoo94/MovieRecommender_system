import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

user_detail = pd.read_csv("u1.base",sep='\t',header=None,encoding="Latin1")
user_detail.columns = ["user_id", "movie_id", "rating", "timestamp"]

movie_detail = pd.read_csv("u.item",sep='|',header=None,encoding="Latin1")
movie_detail = movie_detail[movie_detail.columns[0:2]] 
movie_detail.columns = ["movie_id", "movie_name"]

mean_user_rate = user_detail.groupby(by="user_id",as_index=False)['rating'].mean()
normalised = pd.merge(user_detail,mean_user_rate,on='user_id')
normalised['difference'] = normalised['rating_x'] - normalised['rating_y']

## cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

full_data = pd.pivot_table(normalised,values='difference',index='user_id',columns='movie_id')
full_data_m = full_data.fillna(full_data.mean(axis=0))
# full data user
full_data_u = full_data.apply(lambda row: row.fillna(row.mean()), axis=1)

cosine_table = cosine_similarity(full_data_u)
np.fill_diagonal(cosine_table, 0 )
similarity =pd.DataFrame(cosine_table,index = full_data_u.index)
similarity.columns=full_data.index

def find_n_neighbours(df,n):
    
    df = df.apply(lambda x: pd.Series(x.sort_values(ascending=False)
           .iloc[:n].index, 
          index=['top{}'.format(i) for i in range(1, n+1)]), axis=1)
    return df

Rating_avg = user_detail.astype({"movie_id": str})
Movie_user = Rating_avg.groupby(by = 'user_id')['movie_id'].apply(lambda x:','.join(x))

def score_user(user):
    Movie_seen_by_user = full_data.columns[full_data[full_data.index==user].notna().any()].tolist()
    a = sim_user[sim_user.index == user].values
    b = a.squeeze().tolist()
    d = Movie_user[Movie_user.index.isin(b)]
    l = ','.join(d.values)
    Movie_seen_by_similar_users = l.split(',')
    # non watched movie list
    Movies_under_consideration = list(set(Movie_seen_by_similar_users) - set(list(map(str, Movie_seen_by_user))))
    Movies_under_consideration = list(map(int, Movies_under_consideration))
    score = []
    for item in Movies_under_consideration:
        c = full_data_u.loc[:,item]
        d = c[c.index.isin(b)]
        f = d[d.notnull()]
        avg_user = mean_user_rate.loc[mean_user_rate['user_id'] == user,'rating'].values[0]
        index = f.index.values.squeeze().tolist()
        corr = similarity.loc[user,index]
        fin = pd.concat([f, corr], axis=1)
        fin.columns = ['difference','correlation']
        fin['score']=fin.apply(lambda x:x['difference'] * x['correlation'],axis=1)
        nume = fin['score'].sum()
        deno = fin['correlation'].sum()
        final_score = avg_user + (nume/deno)
        score.append(final_score)
    score_data = pd.DataFrame({'movie_id':Movies_under_consideration,'score':score})
    top_10_recommendation = data.sort_values(by='score',ascending=False).head(10)
    Movie_Name = top_10_recommendation.merge(movie_detail, how='inner', on='movie_id')
    recommend_movies = Movie_Name.iloc[0:10, 2]


    return recommend_movies


sim_user = find_n_neighbours(similarity, 5)
print('Select User id:')
userid = input()
recommendation = score_user(userid)




