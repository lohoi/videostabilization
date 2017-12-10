#!/usr/bin/env python

'''
A failed attempt to use PuLP, a linear program solver package,
to solve Algorithm 1 presented in the paper
'''

from pulp import *
import numpy as np

from read_video import *
import pickle

import csv
from estimate_path import *


def solve_path(F, fc_, vid_height_, vid_width_, crop_ratio_= 0.8):
	print 'solve_path called'
	print 'width: ', vid_width_
	print 'height: ', vid_height_
	print 'fc: ', fc_
	prob = LpProblem('vidstab', pulp.LpMinimize)

	## Parameters
	c1 = np.array([1, 1, 100, 100, 100, 100])
	c2 = np.array([1, 1, 100, 100, 100, 100])
	c3 = np.array([1, 1, 100, 100, 100, 100])

	w1 = 10
	w2 = 1
	w3 = 100;

	# lb = [0.9, -0.1, -0.1, 0.9, -0.1, -0.05]

	# ub = [1.1, 0.1, 0.1, 1.1, 0.1, 0.05]

	# Number of variables
	n = fc_ - 1
	dof = 6 # affine transform

	center_x = int(vid_height_/ 2);
	center_y = int(vid_width_ / 2);
	crop_w = int(vid_width_ * crop_ratio_);
	crop_h = int(vid_height_ * crop_ratio_);


	crop_x = int(center_x - crop_h / 2);
	crop_y = int(center_y - crop_w / 2);
	crop_points = [
					[crop_x, crop_y],
					[crop_x + crop_h, crop_y],
					[crop_x, crop_y + crop_w],
					[crop_x + crop_h, crop_y + crop_w]
				];

	# Slacks
	# c = LpVariable.matrix("c", (c1,c1), 0)
	e1 = LpVariable.matrix('e1', (list(range(n)), list(range(dof))))
	e2 = LpVariable.matrix('e2', (list(range(n)), list(range(dof))))
	e3 = LpVariable.matrix('e3', (list(range(n)), list(range(dof))))
	p = LpVariable.matrix('p', (list(range(n)), list(range(dof))))

	# Objective: minimize c'e
	prob += lpSum(w1 * np.dot(e1, c1) + w2 * np.dot(e2,c2) + w3 * np.dot(e3,c3))
	# prob +=  lpSum([c1[i] * e1[i] for i in range(len(c1))])
	for i in range(n - 3):
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
		temp2 = [e1[i][j] >= reshaped_residuals[j] for j in range(len(reshaped_residuals)) ]

		temp3 = [-1 * e2[i][j] <= reshaped_residuals_t1[j] for j in range(len(reshaped_residuals_t1)) ]
		temp4 = [e2[i][j] >= reshaped_residuals_t1[j] for j in range(len(reshaped_residuals_t1)) ]

		temp5 = [-1 * e3[i][j] <= reshaped_residuals_t2[j] for j in range(len(reshaped_residuals_t2)) ]
		temp6 = [e3[i][j] >= reshaped_residuals_t2[j] for j in range(len(reshaped_residuals_t2)) ]
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

	for i in range(n):
		# proximity points
		prob += p[i][2] >= 0.9
		prob += p[i][5] <= 1.1
		prob += p[i][3] >= -0.1
		prob += p[i][4] <= 0.1
		prob += p[i][3] + p[i][4] >= -0.05
		prob += p[i][3] + p[i][4] <= 0.05
		prob += p[i][2] - p[i][5] >= -0.1
		prob += p[i][2] - p[i][5] <= 0.1

	for i in range(len(crop_points)):
		# inclusion
		for j in range(n):
			temp1 = np.dot(np.array([1, 0, crop_points[i][0], crop_points[i][1], 0, 0]), np.transpose(p[j]))
			prob += 0 <= temp1
			prob += vid_height_ >= temp1

			temp2 = np.dot(np.array([0, 1, 0, 0, crop_points[i][0], crop_points[i][1]]), np.transpose(p[j]))
			prob += 0 <= temp2
			prob += vid_width_ >= temp2
	for t in range(n-3, n):
		for j in range(dof):
			prob += p[t][j] == p[n - 1][j]

	prob.solve()
	pulp.LpStatus[prob.status]

	B = [np.zeros((3,3)) for _ in range(n)]
	for v in prob.variables():
		# put the values into a B_matrix
		if 'p_' in v.name:
			p_name = v.name.split('_')
			name = p_name[0]
			idx = int(p_name[1])
			coeff_idx = p_name[2]

			if coeff_idx == '0':
				# dx
				B[idx][0,2] = v.varValue
			elif coeff_idx == '1':
				# dy
				B[idx][1,2] = v.varValue
			elif coeff_idx == '2':
				# a
				B[idx][0,0] = v.varValue
			elif coeff_idx == '3':
				# b
				B[idx][0,1] = v.varValue
			elif coeff_idx == '4':
				# c
				B[idx][1,0] = v.varValue
			elif coeff_idx == '5':
				# d
				B[idx][1,1] = v.varValue	
			B[idx][2,2] = 1

	# Print the value of the objective
	print("objective=", value(prob.objective))
	pickle.dump(F, open("B_albert.p", "wb"))
	# with open('B_albert.csv', 'w+') as csvfile:
	# 	f_writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
	# 	for b in B:
	# 		f_writer.writerow([b[0,0]] + [b[0,1]] +  [b[0,2]] + [b[0,0]] + [b[1,1]] +  [b[1,2]] + [b[2,0]] + [b[2,1]] +  [b[2,2]])
	print('optimal path returning with length', len(B))
	return B
	# return 0


# filename = '../media/test_vid_eric.mp4'
# vid = read_video(filename)
# # print 'vid.shape', vid.shape
# F = estimate_path(vid, method='NN')
# pickle.dump(F, open("F_a.p", "wb"))
# fc, height, width, rgb = vid.shape
# # F = pickle.load(open("F_a.p", "rb"))
# print "solving path"
# B = solve_path(F,fc,height, width)
# plot_new_path(F,B)