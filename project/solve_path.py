from pulp import *
import numpy as np

from read_video import *
import pickle

from random import *


def solve_path(F_, fc_, vid_height_, vid_width_):
	print 'solve_path called with number of frames: ', fc_
	prob = LpProblem('vidstab', pulp.LpMinimize)

	## Parameters
	c1 = np.array([1, 1, 100, 100, 100, 100])
	c2 = c1
	c3 = c1

	lb = [0.9, -0.1, -0.1, 0.9, -0.1, -0.05]

	ub = [1.1, 0.1, 0.1, 1.1, 0.1, 0.05]

	# Size of the integers involved
	D = 100

	# Number of variables
	n = fc # todo, change this to fc later
	dof = 6

	U = np.zeros((6,6))
	U[2, 0] = 1
	U[5, 3] = 1
	U[3, 1] = 1
	U[4, 2] = 1
	U[2, 4] = 1
	U[5, 4] = -1
	U[3, 5] = 1
	U[4, 5] = 1

	# A vector of n binary variables
	# 	x = LpVariable.matrix("x", list(range(fc)), 0, 1, LpInteger)
	# x = LpVariable.matrix("x", list(range(n)), 0, 1, LpInteger)

	# Slacks
	# c = LpVariable.matrix("c", (c1,c1), 0)
	e1 = LpVariable.matrix('e1', (list(range(n)), list(range(dof))))
	e2 = LpVariable.matrix('e2', (list(range(n)), list(range(dof))))
	e3 = LpVariable.matrix('e3', (list(range(n)), list(range(dof))))
	p = LpVariable.matrix('p', (list(range(n)), list(range(dof))))
	# e3 = 

	# print 'len', len(e1)
	# print 'len2', len(e1[0])
	# # Objective: minimize c'e
	prob += lpSum(np.dot(e1, c1)) + lpSum(np.dot(e2,c2)) + lpSum(np.dot(e3,c3))
	# print prob
	# prob +=  lpSum([c1[i] * e1[i] for i in range(len(c1))]) 
	for i in range(n - 4):
		# constraints
		B = []
		B_t1 = []
		B_t2 = []
		B_t3 = []
		for j in range(3):
			# create a 3d list
			B.append([0,0,0])
			B_t1.append([0, 0, 0])
			B_t2.append([0, 0, 0])
			B_t3.append([0, 0, 0])

		B[0][0] = p[i][2]
		B[0][1] = p[i][3]
		B[0][2] = p[i][0]
		B[1][0] = p[i][4]
		B[1][1] = p[i][5]
		B[1][2] = p[i][1]
		B[2][2] = 1

		B_t1[0][0] = p[i + 1][2]
		B_t1[0][1] = p[i + 1][3]
		B_t1[0][2] = p[i + 1][0]
		B_t1[1][0] = p[i + 1][4]
		B_t1[1][1] = p[i + 1][5]
		B_t1[1][2] = p[i + 1][1]
		B_t1[2][2] = 1

		B_t2[0][0] = p[i + 2][2]
		B_t2[0][1] = p[i + 2][3]
		B_t2[0][2] = p[i + 2][0]
		B_t2[1][0] = p[i + 2][4]
		B_t2[1][1] = p[i + 2][5]
		B_t2[1][2] = p[i + 2][1]
		B_t2[2][2] = 1

		B_t3[0][0] = p[i + 3][2]
		B_t3[0][1] = p[i + 3][3]
		B_t3[0][2] = p[i + 3][0]
		B_t3[1][0] = p[i + 3][4]
		B_t3[1][1] = p[i + 3][5]
		B_t3[1][2] = p[i + 3][1]
		B_t3[2][2] = 1

		int_residuals = np.dot(F[i+1], B_t1) - B
		int_residuals_t1 = np.dot(F[i+2], B_t2) - int_residuals
		int_residuals_t2 = np.dot(F[i+3], B_t3) - 2 * int_residuals_t1 + int_residuals

		reshaped_residuals = []
		reshaped_residuals_t1 = []
		reshaped_residuals_t2 = []

		reshaped_residuals.append(int_residuals[0][2])
		reshaped_residuals.append(int_residuals[1][2])
		reshaped_residuals.append(int_residuals[0][0])
		reshaped_residuals.append(int_residuals[0][1])
		reshaped_residuals.append(int_residuals[1][0])
		reshaped_residuals.append(int_residuals[1][1])

		reshaped_residuals_t1.append(int_residuals[0][2])
		reshaped_residuals_t1.append(int_residuals[1][2])
		reshaped_residuals_t1.append(int_residuals[0][0])
		reshaped_residuals_t1.append(int_residuals[0][1])
		reshaped_residuals_t1.append(int_residuals[1][0])
		reshaped_residuals_t1.append(int_residuals[1][1])

		reshaped_residuals_t2.append(int_residuals[0][2])
		reshaped_residuals_t2.append(int_residuals[1][2])
		reshaped_residuals_t2.append(int_residuals[0][0])
		reshaped_residuals_t2.append(int_residuals[0][1])
		reshaped_residuals_t2.append(int_residuals[1][0])
		reshaped_residuals_t2.append(int_residuals[1][1])

		temp = [-1 * e1[i][j] <= reshaped_residuals[j] for j in range(len(reshaped_residuals)) ]
		temp2 = [e1[i][j] <= reshaped_residuals[j] for j in range(len(reshaped_residuals)) ]

		temp3 = [-1 * e2[i][j] <= reshaped_residuals_t1[j] for j in range(len(reshaped_residuals_t1)) ]
		temp4 = [e2[i][j] <= reshaped_residuals_t1[j] for j in range(len(reshaped_residuals_t1)) ]

		temp5 = [-1 * e3[i][j] <= reshaped_residuals_t2[j] for j in range(len(reshaped_residuals_t2)) ]
		temp6 = [e3[i][j] <= reshaped_residuals_t2[j] for j in range(len(reshaped_residuals_t2)) ]
		for t in range(len(temp)):
			# smoothness constraints
			prob += temp[t]
			prob += temp2[t]
			prob += temp3[t]
			prob += temp4[t]
			prob += temp5[t]
			prob += temp6[t]
			prob += e1[i][0] >= 0
			prob += e1[i][1] >= 0
			prob += e1[i][2] >= 0
			prob += e1[i][3] >= 0
			prob += e1[i][4] >= 0
			prob += e1[i][5] >= 0

			prob += e2[i][0] >= 0
			prob += e2[i][1] >= 0
			prob += e2[i][2] >= 0
			prob += e2[i][3] >= 0
			prob += e2[i][4] >= 0
			prob += e2[i][5] >= 0

			prob += e3[i][0] >= 0
			prob += e3[i][1] >= 0
			prob += e3[i][2] >= 0
			prob += e3[i][3] >= 0
			prob += e3[i][4] >= 0
			prob += e3[i][5] >= 0
	
	for i in range(len(F)):
		# proximity points
		res = np.dot(p[i],U)
		for j in range(len(lb)):
			prob += lb[j] <= res[j]
			prob += ub[j] >= res[j]


	prob.solve()
	# # # Print the value of the variables at the optimum
	for v in prob.variables():
		if 'p' in v.name:
			print(v.name, "=", v.varValue)

	# Print the value of the objective
	print("objective=", value(prob.objective))
	return prob.objective
	# return 0


filename = '../media/test_vid_eric.mp4'
vid = read_video(filename)
fc, height, width, rgb = vid.shape
F = pickle.load(open("F.p", "rb"))
solve_path(F,fc,height, width)