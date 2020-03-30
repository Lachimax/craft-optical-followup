import PyCRAFT.utils as u
import numpy as np
import numpy.random as rand


def value_from_pdf(values, probabilities):
    """
    Produces a pseudorandom number given a custom probability distribution.
    Be warned: quite slow when all probabilities are << 1.
    The basis of this algorithm from:
    https://www.khanacademy.org/computing/computer-programming/programming-natural-simulations/programming-randomness/a/custom-distribution-of-random-numbers

    :param values: numpy array of values to be chosen from.
    :param probabilities: normalised numpy array of probabilities
    :return: value chosen
    """
    if any(probabilities) > 1:
        raise ValueError('All values in the probabilities array must be less than or equal to 1')
    if probabilities.shape != values.shape:
        raise ValueError('The two arrays must be the same length')

    val = False

    while not val:
        # Pick a random number within the range of 'values'
        r1 = rand.uniform(min(values), max(values))

        # Using probabilities array, assign a probability to this value
        idx, _ = u.find_nearest(values, r1)
        p = probabilities[idx]

        # Pick another random num
        r2 = rand.random()
        # print(str(r1) + ' ' + str(p) + ' ' + str(r1))

        if r2 < p:
            return r1


def population_from_pdf(n, x, probabilities):
    pop = np.zeros(n)
    for num in range(n):
        pop[num] = value_from_pdf(values=x, probabilities=probabilities)
    return pop


def normal_curve(x, mu, sigma):
    return (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-(x - mu) ** 2 / (2 * sigma))


def asymmetric_gaussian(x, mu, sigma_plus, sigma_minus):
    dist = np.piecewise(x, [x > mu, x <= mu], [lambda xx: np.exp(-(xx - mu) ** 2 / (2 * sigma_plus ** 2)),
                                               lambda xx: np.exp(-(xx - mu) ** 2 / (2 * sigma_minus ** 2))])
    return dist


def exponential(x, a, c, d):
    return a * np.exp(-c * x) + d
