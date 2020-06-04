users = [
    { "id" : 0, "name" : "Hero"},
    { "id" : 1, "name" : "Dunn"},
    { "id" : 2, "name" : "Sue"},
    { "id" : 3, "name" : "Chi"},
    { "id" : 4, "name" : "Thor"},
    { "id" : 5, "name" : "Clive"},
    { "id" : 6, "name" : "Hicks"},
    { "id" : 7, "name" : "Devin"},
    { "id" : 8, "name" : "Kate"},
    { "id" : 9, "name" : "Klein"}
]

friendship_pairs = [(0,1),(0,2),(1,2),(1,3),(2,3),(3,4),(4,5),(5,6),(5,7),(6,8),(7,8),(8,9)]
print()

# 사용자별로 비어 있는 친구 목록 리스트를 지정하여 딕셔너리를 초기화
friendships = {user["id"]: [] for user in users}
print(f"friendships : {friendships}")
print()

# friendship_pairs 안의 쌍을 차례대로 살펴보면서 딕셔너리 안에 추가
for i, j in friendship_pairs:
    friendships[i].append(j) # j를 사용자 i의 친구로 추가
    friendships[j].append(i) # i를 사용자 j의 친구로 추가
print(f"appended friendships : {friendships}")
print()

def number_of_friends(user):
    """user의 친구는 몇멍일까?"""
    user_id = user["id"]
    friend_ids = friendships[user_id]
    return len(friend_ids)

total_connections = sum(number_of_friends(user) for user in users)   
print(f"total_connections : {total_connections}") #24
print()

num_users = len(users)                              # 총 사용자 리스트의 길이
avg_connections = total_connections / num_users     # 24 / 10 == 2.4

# (user_id, number_of_friends)로 구성된 리스트 생성
num_friends_by_id = [(user["id"], number_of_friends(user)) for user in users]

num_friends_by_id.sort(                             # .sort()
    key=lambda id_and_friends: id_and_friends[1],   # num_friends 기준
    reverse=True)                                   # 내림차순 정렬
print(f"sorted num_friends_by_id : {num_friends_by_id}") # (user_id, num_friends)
print()

# foaf = friends of a friend
def foaf_ids_bad(user):
    return [foaf_id
            for friend_id in friendships[user["id"]]
            for foaf_id in friendships[friend_id]]
print(foaf_ids_bad(users[0])) # Hero's foaf includes himself
print(foaf_ids_bad(users[2])) # Chi's foaf includes three '2's because he knows 3 friends 
print()

print(friendships[0])
print(friendships[1])
print(friendships[2])
print()

from collections import Counter

def friends_of_friends(user):
    user_id = user["id"]
    return Counter(
        foaf_id
        for friend_id in friendships[user_id]
        for foaf_id in friendships[friend_id]
        if foaf_id != user_id
        and foaf_id not in friendships[user_id]
    )

print(friends_of_friends(users[3]))
print()

interests = [
    (0, "Hadoop"), (0, "Big Data"), (0, "HBase"), (0, "Java"),
    (0, "Spark"), (0, "Storm"), (0, "Cassandra"),
    (1, "NoSQL"), (1, "MongoDB"), (1, "Cassandra"), (1, "HBase"),
    (1, "Postgres"), (2, "Python"), (2, "scikit-learn"), (2, "scipy"),
    (2, "numpy"), (2, "statsmodels"), (2, "pandas"), (3, "R"), (3, "Python"),
    (3, "statistics"), (3, "regression"), (3, "probability"),
    (4, "machine learning"), (4, "regression"), (4, "decision trees"),
    (4, "libsvm"), (5, "Python"), (5, "R"), (5, "Java"), (5, "C++"),
    (5, "Haskell"), (5, "programming languages"), (6, "statistics"),
    (6, "probability"), (6, "mathematics"), (6, "theory"),
    (7, "machine learning"), (7, "scikit-learn"), (7, "Mahout"),
    (7, "neural networks"), (8, "neural networks"), (8, "deep learning"),
    (8, "Big Data"), (8, "artificial intelligence"), (9, "Hadoop"),
    (9, "Java"), (9, "MapReduce"), (9, "Big Data")
]

# function to return all users id who have a specific interest
def data_scientists_who_like(target_interest):
    return [user_id
            for user_id, user_interest in interests
            if user_interest == target_interest]

# i'd rater make index
from collections import defaultdict
