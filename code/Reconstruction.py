import os
import polyscope as ps
import numpy as np
from skimage import measure
from ReconstructionFunctions import load_off_file, compute_RBF_weights, evaluate_RBF


if __name__ == '__main__':
    ps.init()

    inputPointNormals, _ = load_off_file(os.path.join('data', 'bunny-1000.off'))
    inputPoints = inputPointNormals[:, 0:3]
    inputNormals = inputPointNormals[:, 3:6]

    # normalizing point cloud to be centered on [0,0,0] and between [-0.9, 0.9]
    inputPoints -= np.mean(inputPoints, axis=0)
    min_coords = np.min(inputPoints, axis=0)
    max_coords = np.max(inputPoints, axis=0)
    scale_factor = 0.9 / np.max(np.abs(inputPoints))
    inputPoints = inputPoints * scale_factor

    ps_cloud = ps.register_point_cloud("Input points", inputPoints)
    ps_cloud.add_vector_quantity("Input Normals", inputNormals)

    # Parameters
    gridExtent = 1 #the dimensions of the evaluation grid for marching cubes
    res = 50 #the resolution of the grid (number of nodes)

    # Generating and registering the grid
    gridDims = (res, res, res)
    bound_low = (-gridExtent, -gridExtent, -gridExtent)
    bound_high = (gridExtent, gridExtent, gridExtent)
    ps_grid = ps.register_volume_grid("Sampled Grid", gridDims, bound_low, bound_high)

    X, Y, Z = np.meshgrid(np.linspace(-gridExtent, gridExtent, res),
                          np.linspace(-gridExtent, gridExtent, res),
                          np.linspace(-gridExtent, gridExtent, res), indexing='ij')

    #the list of points to be fed into evaluate_RBF
    xyz = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

    ##########################
    ## you code of computation and evaluation goes here
    ##
    ##
    
    def polyharmonic(r):
        result = np.power(r, 3)
        return result
    
    def biharmonic(r):
        return r
    
    def Wendland(r):
        beta = 0.05
        result = (1 / 12) * (np.maximum((1 - beta * r), 0) ** 3) * (1-3 * beta * r)
        return result
    

    l=-1
    RBFFunction = polyharmonic
    epsilon = 0.005
    useOffPoints = True
    sparsify = False
    
    RBFCentreIndices = [358, 324, 566, 240, 365, 764, 35, 151, 358, 579, 158, 364, 347, 989, 762, 929, 893, 467, 127, 576, 466, 662, 783, 764, 780, 270, 165, 832, 240, 64, 647, 748, 433, 869, 623, 401, 788, 195, 361, 271, 691, 231, 666, 354, 986, 709, 58, 851, 769, 232, 496, 762, 499, 442, 300, 961, 608, 445, 83, 723, 398, 413, 590, 785, 445, 889, 810, 800, 501, 708, 78, 186, 8, 353, 573, 358, 507, 46, 559, 867, 963, 738, 829, 117, 641, 94, 386, 702, 154, 541, 524, 971, 492, 913, 488, 980, 244, 628, 635, 239, 652, 114, 142, 586, 673, 885, 693, 117, 718, 638, 845, 691, 794, 774, 167, 527, 725, 586, 432, 454, 939, 514, 637, 282, 424, 748, 956, 357, 740, 623, 723, 122, 826, 302, 290, 11, 231, 85, 352, 243, 358, 608, 945, 857, 393, 551, 316, 435, 846, 40, 156, 699, 948, 325, 624, 80, 114, 226, 868, 648, 547, 16, 272, 738, 163, 866, 220, 931, 84, 283, 157, 346, 123, 743, 974, 100, 16, 107, 278, 313, 269, 841, 726, 432, 55, 70, 285, 489, 976, 52, 283, 113, 871, 670, 657, 980, 666, 850, 383, 542, 618, 710, 450, 48, 295, 778, 562, 177, 932, 876, 85, 255, 923, 790, 260, 293, 715, 112, 809, 589, 258, 108, 310, 308, 583, 445, 672, 768, 898, 16, 717, 979, 161, 522, 101, 694, 83, 4, 71, 882, 300, 597, 139, 455, 667, 492, 89, 232, 15, 868, 25, 258, 268, 21, 628, 461, 4, 919, 838, 239, 178, 105, 235, 133, 188, 700, 306, 340, 922, 266, 120, 543, 972, 449, 291, 248, 239, 41, 154, 727, 398, 480, 858, 461, 509, 979, 114, 989, 933, 119, 804, 263, 296, 724, 17, 702, 268, 667, 587, 776, 868, 183, 910, 605, 478, 674, 858, 663, 241, 738, 163, 358, 916, 535, 112, 288, 461, 413, 127, 439, 803, 991, 572, 551, 819, 324, 75, 304, 452, 465, 749, 618, 872, 718]

    # RBFCentreIndices = []
        
    w, centers, a = compute_RBF_weights(inputPoints, inputNormals, RBFFunction, epsilon = epsilon, RBFCentreIndices=RBFCentreIndices, useOffPoints=useOffPoints, sparsify=sparsify, l=l)
    
    RBFValues = evaluate_RBF(xyz, centers, RBFFunction, w, l=l, a=a)
    ##
    ##
    ##
    #########################
    #fitting to grid shape again
    RBFValues = np.reshape(RBFValues, X.shape)

    # Registering the grid representing the implicit function
    ps_grid.add_scalar_quantity("Implicit Function", RBFValues, defined_on='nodes',
                                datatype="standard", enabled=True)

    # Computing marching cubes and realigning result to sit on point cloud exactly
    vertices, faces, _, _ = measure.marching_cubes(RBFValues, spacing=(
        2.0 * gridExtent / float(res - 1), 2.0 * gridExtent / float(res - 1), 2.0 * gridExtent / float(res - 1)),
                                                   level=0.0)
    vertices -= gridExtent
    ps.register_surface_mesh("Marching-Cubes Surface", vertices, faces)

    ps.show()