# Python3 implementation of Finding 
# Length of Longest Common Substring 
import gc
import datetime
# Returns length of longest common 
# substring of X[0..m-1] and Y[0..n-1] 
def LCSubStr(X, Y, m, n): 
	
	# Create a table to store lengths of 
	# longest common suffixes of substrings. 
	# Note that LCSuff[i][j] contains the 
	# length of longest common suffix of 
	# X[0...i-1] and Y[0...j-1]. The first 
	# row and first column entries have no 
	# logical meaning, they are used only 
	# for simplicity of the program. 
	
	# LCSuff is the table with zero 
	# value initially in each cell 
	LCSuff = [[0 for k in range(n+1)] for l in range(m+1)] 
	
	# To store the length of 
	# longest common substring 
	result = 0
	# Following steps to build 
	# LCSuff[m+1][n+1] in bottom up fashion 
	for i in range(m + 1): 
		for j in range(n + 1): 
			if (i == 0 or j == 0): 
				LCSuff[i][j] = 0
			elif (X[i-1] == Y[j-1]): 
				LCSuff[i][j] = LCSuff[i-1][j-1] + 1
				result = max(result, LCSuff[i][j]) 
			else: 
				LCSuff[i][j] = 0
	
	del LCSuff
	# gc.collect()
	return result 

# Driver Program to test above function 
X = [1, 2, 3, 4]
Y = [2, 3, 4, 10]
# C = "123Site"

a = []
# for i in range():
a.append(X)
a.append(Y)

print("a has element", len(a)*(len(a)+1)*0.5)
max_common_len = 0

s_time = datetime.datetime.now()
for i in range(len(a)):
	for j in range(i+1, len(a)):
		X = a[i]
		
		Y = a[j]
		
		m = len(X) 
		n = len(Y) 

		common_len = LCSubStr(X, Y, m, n)
		if common_len > max_common_len:
			max_common_len = common_len

		# print('Length of Longest Common Substring is', 
					# LCSubStr(X, Y, m, n)) 
e_time = datetime.datetime.now()
print("max_common_len", max_common_len)
print("duration", e_time-s_time)
# This code is contributed by Soumen Ghosh 
