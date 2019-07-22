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
'''

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

warm_data = []
warm_data_items = []
cold_data = []
cold_users = []
warm_users = []

cold_users_split = 0.3
num_cold_users = int(cold_users_split*numOfUsers)

print "Reading file.............."
data = np.genfromtxt (netflixDir+filename, delimiter=delim)
print "...............File reading complete!"

print "Processing users ..."
arr = np.array(data)
allUsers = arr[:, userColumn]
uniqueUsers = set(allUsers)
cold_users = np.random.choice(list(uniqueUsers), num_cold_users)
print "Processing users ... COMPLETE!"

for line in data:
	print line[itemColumn]
	if line[userColumn] in cold_users: # Known cold user
		cold_data.append([line[itemColumn], line[userColumn], line[ratingColumn]])
	else:	
        warm_data.append([line[itemColumn], line[userColumn], line[ratingColumn]])
        warm_data_items.append(line[itemColumn])
		warm_users.append(line[userColumn])	
	print "Processed movie: ", (line[itemColumn])

## check if there exists some item that is only rated by cold_users, if so, remove them
## because the orders of itemColumn and UserColumn have been swapped(0 to 1 and 1 to 0)
## to avoid possible confusion, we use val[0] to represent items. Should Not use val[itemColumn]
for idx, val in list(enumerate(cold_data)):
    if val[0] not in warm_data_items:
        print('the item', val[0], 'is only rated by cold users')
        cold_data = np.delete(cold_data, idx)
        print('discarded item rated only by cold users')

sio.savemat(netflixDir+'data_withoutrat_randcold2.mat', {'warm':warm_data, 'cold':cold_data})
print "Saved file as .mat ..........."
print len(set(cold_users))