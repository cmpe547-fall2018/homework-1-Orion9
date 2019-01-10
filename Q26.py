import numpy as np

if __name__ == '__main__':
    x = np.array([1, 2])
    y = np.array([-1, 0, 5])
    p_xy = np.array([[0.3, 0.3, 0.0], [0.1, 0.2, 0.1]])

    # Marginalize on x and y
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)

    # Since p(x|y) = p(x,y)/p(y) and p(y|x) = p(x,y)/p(x)
    p_x_given_y = p_xy / p_y
    p_y_given_x = p_xy.T / p_x

    # First part
    expectation_x = np.sum(x * p_x)
    expectation_y = np.sum(y * p_y)
    expectation_x_given_y = np.sum(y * p_x_given_y)
    expectation_y_given_x = np.sum(x * p_y_given_x)

    diff_x = np.array([i - expectation_x for i in x]).reshape(x.shape[0], 1)
    diff_y = np.array([i - expectation_y for i in y]).reshape(1, y.shape[0])
    cov_x_y = np.sum(np.dot(diff_x, diff_y) * p_xy)

    print("<x> = {0}".format(expectation_x))
    print("<y> = {0}".format(expectation_y))
    print("<x | y> = {0}".format(expectation_x_given_y))
    print("<y | x> = {0}".format(expectation_y_given_x))
    print("Cov[x, y] = {0}".format(cov_x_y))

    # Second part
    joint_entropy = -1 * np.sum(np.log(p_xy + 1e-150) * p_xy)
    print("H[x, y] = {0}".format(joint_entropy))

    # Third part
    entropy_x = -1 * np.sum(np.log(p_x) * p_x)
    entropy_y = -1 * np.sum(np.log(p_y) * p_y)
    print("H[x] = {0}".format(entropy_x))
    print("H[y] = {0}".format(entropy_y))

    # Fourth part
    entropy_y_given_x = -1 * np.sum(np.log(p_y_given_x + 1e-150).T * p_xy)
    entropy_x_given_y = -1 * np.sum(np.log(p_x_given_y + 1e-150) * p_xy)
    print("H[y|x] = {0}".format(entropy_y_given_x))
    print("H[x|y] = {0}".format(entropy_x_given_y))

    # Fifth part
    i_xy = entropy_x - entropy_x_given_y
    print("I[x, y] = {0}".format(i_xy))
