'''
Input: Filename consisting of tuples of item, user, ratings
Output: .mat file splitting users randomly into cold and warm users, along with all their ratings in the order [itemid userid rating]
'''

import scipy.io as sio
import numpy as np
import random

print "Initializing ..."
'''
### Dataset values for Netflix
netflixDir = ""
numOfUsers = 480189
numOfItems = 17770
userIndex = 2649429
itemColumn = 0
userColumn = 1
ratingColumn = 2
delim = "," # Delimiter
filename = "ratings.txt"

### Dataset values for Movielens 1M
netflixDir = ""
numOfUsers = 6040
numOfItems = 3952
userIndex = 6040
itemColumn = 1
userColumn = 0
ratingColumn = 2
delim = "::"
filename = "ratings.dat"


### Dataset values for Movielens 100K
netflixDir = ""
numOfUsers = 943
numOfItems = 1682
userIndex = 943
itemColumn = 1
userColumn = 0
ratingColumn = 2
delim = ","
filename = "u.csv"

### Dataset values for Movielens 20M
netflixDir = ""
numOfUsers = 138493
numOfItems = 131262 #27278
userIndex = 138493
itemColumn = 1
userColumn = 0
ratingColumn = 2
delim = ","
filename = "ratings.csv"


### Dataset values for Epinions
netflixDir = ""
numOfUsers = 49290
numOfItems = 139738
userIndex = 49290
itemColumn = 1
userColumn = 0
ratingColumn = 2
delim = " "
filename = "ratings_data.txt"
'''
### Dataset values for goodread_interaction
netflixDir = "./data/goodread_interaction/"
# numOfUsers = 808749
# numOfUsers = 16174
numOfUsers = 876145
numOfItems = 1561465
userIndex = 808749
itemColumn = 1
userColumn = 0
ratingColumn = 3
delim = ","
filename = "goodreads_interactions.csv"


warm_data = []
cold_data = []
cold_users = []
warm_items = []
# warm_users = []
##nSample = 4000000

cold_users_split = 0.3
num_cold_users = int(cold_users_split * numOfUsers)

print("Reading file..............")
data = np.genfromtxt(netflixDir + filename, delimiter=delim)
print("...............File reading complete!")
data = np.delete(data, 0, axis=0)   # comment this line out if not processing goodread_interaction
print("the shape of data is", data.shape)

# data = data[np.random.choice(data.shape[0], nSample, replace=False), :]
# print("the shape of sampled data is", data.shape)
# print("..............Sample complete!")

print("Processing users ...")
arr = np.array(data)
allUsers = arr[:, userColumn]
Userlist = list(allUsers)
uniqueUsers = set(allUsers)
print("there are", len(uniqueUsers), "unique users")
cold_users = np.random.choice(list(uniqueUsers), num_cold_users)
print("Processing users ... COMPLETE!")
'''
for line in data:
    if line[userColumn] not in cold_users:
        warm_items.append(line[itemColumn])
print("Warm items done!")

for line in data:
    if line[userColumn] not in cold_users:
        warm_data.append([line[itemColumn], line[userColumn], line[ratingColumn]])
    elif (line[userColumn] in cold_users) and (line[itemColumn] in warm_items):
        cold_data.append([line[itemColumn], line[userColumn], line[ratingColumn]])
print("Processed Warm/Cold Done!")
'''
for line in data:
    # print(line[itemColumn])
    if line[userColumn] in cold_users:  # Known cold user
        cold_data.append([line[itemColumn], line[userColumn], line[ratingColumn]])
    else:
        warm_data.append([line[itemColumn], line[userColumn], line[ratingColumn]])
        # warm_users.append(line[userColumn])
    # print("Processed movie: ", (line[itemColumn]))

print("processed warm, done!")
warm_data = np.asarray(warm_data)


# check if there exists some item that is only rated by cold_users, if so, remove them
# because the orders of itemColumn and UserColumn have been swapped(0 to 1 and 1 to 0)
# to avoid possible confusion, we use val[0] to represent items. Should Not use val[itemColumn]
def determine(x):
    print('processing original cold_data: ' + str(x))
    if x[0] in warm_data[:, 0]:
        return True
    else:
        return False


print('the shape of warm_array is', warm_data.shape)

cold_data = [x for x in cold_data if determine(x)]
print('discarded items rated only by cold users, done!')

warm_data = np.asarray(warm_data)
cold_data = np.asarray(cold_data)

sio.savemat(netflixDir+'data_withoutrat_randcold2.mat', {'warm':warm_data, 'cold':cold_data})
print "Saved file as .mat ..........."
print len(set(cold_users))
