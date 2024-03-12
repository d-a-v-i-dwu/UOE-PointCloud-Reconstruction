import os
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import lstsq, lu_factor, lu_solve
from functools import partial
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

def load_off_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the vertices and faces from the OFF file
    num_vertices, num_faces, _ = map(int, lines[1].split())

    vertices = np.array([list(map(float, line.split())) for line in lines[2:2 + num_vertices]])
    faces = np.array([list(map(int, line.split()))[1:] for line in lines[2 + num_vertices:]])

    return vertices, faces



def generate_powers(l):
    i, j, k = np.meshgrid(range(l + 1), range(l + 1), range(l + 1), indexing='ij')
    mask = i + j + k <= l
    powers = list(zip(i[mask], j[mask], k[mask]))
    return powers


def compute_power_matrix(coordinate_points, powers):
    power_matrix = np.zeros((len(coordinate_points), len(powers)))
    
    for i, coord in enumerate(coordinate_points):
        for j, power in enumerate(powers):
            result = 1
            for k, p in enumerate(power):
                result *= coord[k] ** p
            power_matrix[i][j] = result
    
    return power_matrix


def generate_matrix_Q(coordinate_points, l):
    powers = generate_powers(l)
    q_matrix = compute_power_matrix(coordinate_points, powers)
    return q_matrix, powers


def calculate_global_polynomial_equation(coordinate_points, l, a, b):
    q_matrix, powers = generate_matrix_Q(coordinate_points, l)
    
    L = len(powers)
    zeroes = np.zeros((L, L))
    q_matrix_t = np.transpose(q_matrix)
    
    b_zeroes = np.zeros(L)
    new_b = np.concatenate((b, b_zeroes))
    
    
    block_matrix = np.block([[a,q_matrix], [q_matrix_t, zeroes]])

    lu, piv = lu_factor(block_matrix)
    w_with_a = lu_solve((lu, piv), new_b)

    new_w = w_with_a[:q_matrix.shape[0]]
    new_a = w_with_a[q_matrix.shape[0]:]
    
    return new_w, new_a



def compute_RBF_weights(inputPoints, inputNormals, RBFFunction, epsilon, RBFCentreIndices=[], useOffPoints=True,
                        sparsify=False, l=-1):
    
    if len(RBFCentreIndices) > 0:
        inputPoints = inputPoints[RBFCentreIndices]
        inputNormals = inputNormals[RBFCentreIndices]
            
    coordinate_points = inputPoints
    
    b = np.repeat(
        [0,
        epsilon,
        -epsilon],
        coordinate_points.shape[0]
    )
    
    outside_points = coordinate_points + epsilon * inputNormals
    inside_points = coordinate_points - epsilon * inputNormals
    coordinate_points = np.concatenate((coordinate_points, outside_points, inside_points))

    if useOffPoints:
        RBFCentres = coordinate_points
    else:
        RBFCentres = inputPoints
    
    a = cdist(coordinate_points, RBFCentres)
    
    a = RBFFunction(a)
    
    if sparsify:
        a, w = compute_sparse_weights(coordinate_points, RBFCentres, RBFFunction, b)
    else:
        if l > -1:
            w, a = calculate_global_polynomial_equation(coordinate_points, l, a, b)
        elif not useOffPoints or len(RBFCentreIndices) > 0:
            w, _, _, _ = lstsq(a, b)
            a = []
        else:
            lu, piv = lu_factor(a)
            w = lu_solve((lu, piv), b)

    return w, RBFCentres, a


def evaluate_RBF(xyz, centres, RBFFunction, w, l=-1, a=[]):

    distance_between_points = cdist(xyz, centres, 'euclidean')
    rbf_evaluated_points = RBFFunction(distance_between_points)
    values = np.dot(rbf_evaluated_points, w)
    
    if l > -1:
        q_matrix, _ = generate_matrix_Q(xyz, l)
        polynomial_coefficients = q_matrix @ a
        values += polynomial_coefficients
    
    return values


# Seciton 4 extension
def compute_sparse_weights(coordinate_points, RBFCentres, RBFFunction, b):
    num_centers = coordinate_points.shape[0] / 3
    a = lil_matrix((num_centers * 3, num_centers * 3))

    for i in range(num_centers):
        distances = cdist(coordinate_points[i], RBFCentres)
        a[i] = RBFFunction(distances)

    a = csr_matrix(a)
    w = spsolve(a, b)

    return a, w


def polyharmonic(r):
    result = np.power(r, 3)
    return result

def biharmonic(r):
    return r

def wendland(r, beta):
    result = (1 / 12) * (np.maximum((1 - beta * r), 0) ** 3) * (1-3 * beta * r)
    return result

Wendland = partial(wendland, beta=0.05)