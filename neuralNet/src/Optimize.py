import numpy as np
def fminimize(cost, theta, grad, args=(), max_iters = 10, alpha = 1e-4, epsilon = 1e-4):

	prev_cost = cost(theta, *args)
	cur_step = 1
	cur_theta = theta
	J = [prev_cost]
	min_not_pos_step = 0

	while cur_step < max_iters:

		cur_step += 1
		
		new_theta = cur_theta - alpha * grad(theta, *args).reshape((-1, 1))
		new_cost = cost(new_theta, *args)
		print("cur_step: {!s} | prev_cost: {!s} | cur_cost: {!s} | alpha: {!s}".format(cur_step, prev_cost, new_cost, alpha))

		if prev_cost > new_cost:

			J.append(new_cost)
			cur_theta = new_theta

			if np.abs(prev_cost - new_cost) < epsilon:
				break

			prev_cost = new_cost
		
		elif prev_cost < new_cost:
			alpha *= 0.1

			#min_not_pos_step += 1

		"""if min_not_pos_step > max_iters / 2:
				raise ValueError("Minimization not possible with these params")"""

	print("Done!")
	return cur_theta.reshape((-1, 1)), np.array(J)

def normalize(X):
	return X - X.mean(axis = 0)

def norm_feature_scale(X):
	return (X - X.mean(axis = 0)) / X.std(axis = 0)