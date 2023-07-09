import numpy as np
from sklearn.svm import LinearSVC

# You are allowed to import any submodules of sklearn as well e.g. sklearn.svm etc
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES OTHER THAN SKLEARN WILL RESULT IN A STRAIGHT ZERO

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_predict etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

def train_preprocessor(Z):
	X = Z[:, 0:R]
	y = 0
	xorro_1 = Z[:, R : R + S]
	xorro_2 = Z[:, R + S : R + 2 * S]
	y = Z[:, R + 2 * S : R + 2 * S + 1].flatten()
	xorro_1 = np.sum(xorro_1 * np.power(2, np.arange(S)[::-1]), axis=1).reshape(1,-1).flatten().astype(np.uint8)
	xorro_2 = np.sum(xorro_2 * np.power(2, np.arange(S)[::-1]), axis=1).reshape(1,-1).flatten().astype(np.uint8)

	for i in range(len(xorro_2)):
		if (xorro_2[i] > xorro_1[i]):
			xorro_1[i], xorro_2[i] = xorro_2[i], xorro_1[i]
			y[i] = 1 - y[i]			
	return X, xorro_1, xorro_2, y

def test_preprocessor(Z):
	X = Z[:, 0:R]
	xorro_1 = Z[:, R : R + S]
	xorro_2 = Z[:, R + S : R + 2 * S]
	xorro_1 = np.sum(xorro_1 * np.power(2, np.arange(S)[::-1]), axis=1).reshape(1,-1).flatten().astype(np.uint8)
	xorro_2 = np.sum(xorro_2 * np.power(2, np.arange(S)[::-1]), axis=1).reshape(1,-1).flatten().astype(np.uint8)		
	return X, xorro_1, xorro_2


def train_test_split(X, xorro_1, xorro_2, y, test_size=0.2):
    # splitting the data into training and testing set
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    xorro_1_train, xorro_1_test = xorro_1[:split_index], xorro_1[split_index:]
    xorro_2_train, xorro_2_test = xorro_2[:split_index], xorro_2[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, xorro_1_train, xorro_1_test, xorro_2_train, xorro_2_test, y_train, y_test

num_xorros = 16

################################
# Non Editable Region Starting #
################################
R=64
S=4


def my_fit( Z_trn ):
################################
#  Non Editable Region Ending  #
################################
	classifiers = np.empty((num_xorros, num_xorros), dtype=type(LinearSVC(loss='squared_hinge')))
	X, xorro_1, xorro_2, y = train_preprocessor(Z_trn)
	for i in range(num_xorros):
		for j in range(i):
			clf = LinearSVC( loss = "squared_hinge" )
			mask_1 = xorro_1==i
			mask_2 = xorro_2==j
			mask = mask_1&mask_2
			X_trn = X[mask]
			y_trn = y[mask]
			# clf.fit(trn_data[i,j,:count_trn[i,j],:-1],trn_data[i,j,:count_trn[i,j],-1] )
			clf.fit(X_trn, y_trn)
			classifiers[i,j]=clf

	# Use this method to train your model using training CRPs
	# The first 64 columns contain the config bits
	# The next 4 columns contain the select bits for the first mux
	# The next 4 columns contain the select bits for the second mux
	# The first 64 + 4 + 4 = 72 columns constitute the challenge
	# The last column contains the response
	
	return classifiers				# Return the trained model


################################
# Non Editable Region Starting #
################################
def my_predict( X_tst ,linear_models):
################################
#  Non Editable Region Ending  #
################################
	X,x1,x2 = test_preprocessor(X_tst)
	y_out = np.zeros(X.shape[0])
	for i in range(num_xorros):
		for j in range(i):
			model = linear_models[i,j]
			mask1 = x1 == i
			mask2 = x2 == j
			mask = mask1&mask2
			X_ij = X[mask]
			y_ij = model.predict(X_ij)
			y_out[mask] = y_ij
			mask3 = x1 == j
			mask4 = x2 == i
			mask = mask3&mask4
			X_ji = X[mask]
			y_ji = model.predict(X_ji)
			y_out[mask] = 1 - y_ji
	return y_out

