
from surprise import SVD, Dataset, Reader # type: ignore
from surprise.model_selection import train_test_split # type: ignore
from surprise import accuracy # type: ignore
import pandas as pd


data = {
    'user': [1, 1, 1, 2, 2, 3, 3, 4, 4],
    'item': ['The Matrix', 'Titanic', 'The Godfather', 'Titanic', 'Inception', 'The Matrix', 'Inception', 'The Godfather', 'Inception'],
    'rating': [5, 3, 4, 4, 5, 2, 5, 5, 3]
}


df = pd.DataFrame(data)


reader = Reader(rating_scale=(1, 5))  
dataset = Dataset.load_from_df(df[['user', 'item', 'rating']], reader)


trainset, testset = train_test_split(dataset, test_size=0.2)


model = SVD()


model.fit(trainset)


predictions = model.test(testset)


rmse = accuracy.rmse(predictions)
print(f"RMSE (Root Mean Squared Error): {rmse}")


user_id = 2  
item_id = 'The Matrix'  
predicted_rating = model.predict(user_id, item_id).est
print(f"Predicted rating for user {user_id} on '{item_id}': {predicted_rating}")


def get_top_n_recommendations(predictions, n=3):
    top_n = {}
    
    for uid, iid, true_r, est, _ in predictions:
        if uid not in top_n:
            top_n[uid] = []
        top_n[uid].append((iid, est))
    
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)  
        top_n[uid] = user_ratings[:n]  
    
    return top_n


top_n_recommendations = get_top_n_recommendations(predictions, n=3)


for user_id, user_ratings in top_n_recommendations.items():
    print(f"Top recommendations for User {user_id}:")
    for item, rating in user_ratings:
        print(f"  - {item} (Predicted rating: {rating})")
