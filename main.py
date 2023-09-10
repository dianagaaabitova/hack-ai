import numpy as np
import pandas as pd
video_df = pd.read_parquet('videos.parquet')
video_df.drop(['season'], axis=1, inplace=True)
player_starts_train_df = pd.read_csv('small_player_starts_train.csv')
player_starts_train_df.drop(['is_autorized'], axis=1, inplace=True)
player_starts_train_df.drop_duplicates(subset=['user_id', 'item_id'], keep='first')
merged_df = pd.merge(player_starts_train_df, video_df, on='item_id', how='left')
def compare_duration_watch_time(row):
    if row['duration'] < 5 * 60 and row['watch_time'] > 30:
        return 1
    elif row['duration'] >= 5 * 60 and row['watch_time'] > 0.25 * row['duration']:
        return 1
    else:
        return 0
dataset=merged_df
print(dataset)
dataset["watch_time"] = dataset.apply(compare_duration_watch_time, axis=1)
dataset['watch_time'].unique()
data=dataset
data = data.groupby('item_id').agg({'watch_time':'sum', 'video_title':'min', 'publicated':'min', 'channel_sub':'max'}).reset_index()
data=data.sort_values(by=['publicated','watch_time', 'channel_sub'], ascending=[False, False, False])
print(data)
data = data[data['watch_time'] > 30]
print(data)
data.head(10)
dataset = dataset.groupby(['user_id', 'item_id']).agg({'watch_time':'sum'}).reset_index()
dataset = dataset[dataset['watch_time'] > 0]
user_item_matrix = dataset.pivot_table(index = 'item_id', columns = 'user_id', values = 'watch_time',aggfunc='sum')
print(user_item_matrix)
user_item_matrix.fillna(0, inplace = True)
datas=merged_df
datas["watch_time"] = datas.apply(compare_duration_watch_time, axis=1)
users_votes = merged_df.groupby('user_id')['watch_time'].agg('count')
movies_votes = merged_df.groupby('item_id')['watch_time'].agg('count')
user_mask = users_votes[users_votes > 10].index
movie_mask = movies_votes[movies_votes > 10].index
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
csr_data = csr_matrix(user_item_matrix.values)
user_item_matrix = user_item_matrix.rename_axis(None, axis = 1).reset_index()
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)
recommendations = int(input('Количество рекомендаций: '))
id = str(input('Введите айди: '))
filtered_df = dataset[(dataset['user_id'] == id) & (dataset['watch_time'] != 0)]
# Получите 'item_id' из отфильтрованных строк
if not filtered_df.empty:
    search_word = filtered_df.iloc[0]['item_id']
else:
    recom_list = []

    # теперь в цикле будем поочередно проходить по кортежам
    for ind_dist in data.head(10):
        # искать movieId в матрице предпочтений
        recom_list.append({'Title': ind_dist['item_id']})

    # индекс будем начинать с 1, как и положено рейтингу
    recom_df = pd.DataFrame(recom_list, index=range(1, recommendations + 1))
    print(recom_df)
    movie_search = user_item_matrix[user_item_matrix['item_id'].str.contains(search_word)]
    print(movie_search)
    # вариантов может быть несколько, для простоты всегда будем брать первый вариант
    # через iloc[0] мы берем первую строку столбца ['movieId']
    movie_id = movie_search.iloc[0]['item_id']
    print(movie_id)

    # далее по индексу фильма в датасете movies найдем соответствующий индекс
    # в матрице предпочтений
    movie_id = user_item_matrix[user_item_matrix['item_id'] == movie_id].index[0]
    distances, indices = knn.kneighbors(csr_data[movie_id], n_neighbors=recommendations + 1)

    # уберем лишние измерения через squeeze() и преобразуем массивы в списки с помощью tolist()
    indices_list = indices.squeeze().tolist()
    distances_list = distances.squeeze().tolist()

    # далее с помощью функций zip и list преобразуем наши списки
    indices_distances = list(zip(indices_list, distances_list))

    # в возрастающем порядке reverse = False
    indices_distances_sorted = sorted(indices_distances, key=lambda x: x[1], reverse=False)

    # и убрать первый элемент с индексом 901 (потому что это и есть "Матрица")
    indices_distances_sorted = indices_distances_sorted[1:]

    # создаем пустой список, в который будем помещать название фильма и расстояние до него
    recom_list = []

    # теперь в цикле будем поочередно проходить по кортежам
    for ind_dist in indices_distances_sorted:
        # искать movieId в матрице предпочтений
        matrix_movie_id = user_item_matrix.iloc[ind_dist[0]]['item_id']

        # выяснять индекс этого фильма в датафрейме movies
        id = video_df[video_df['item_id'] == matrix_movie_id].index

        # брать название фильма и расстояние до него
        # title = video_df.iloc[id]['video_title'].values[0]
        title = video_df[video_df['item_id'] == matrix_movie_id].index
        dist = ind_dist[1]

        # помещать каждую пару в питоновский словарь
        # который, в свою очередь, станет элементом списка recom_list
        recom_list.append({'Title': matrix_movie_id, 'Distance': dist})

    # индекс будем начинать с 1, как и положено рейтингу
    recom_df = pd.DataFrame(recom_list, index=range(1, recommendations + 1))
    print(recom_df)
    metrics = Experiment(test, {NDCG(): K, HitRate(): K})
    metrics.add_result("knn", recs)
