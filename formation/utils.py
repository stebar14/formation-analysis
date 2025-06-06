import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import seaborn as sns
import glob
from scipy.spatial import ConvexHull
from scipy.optimize import linear_sum_assignment
from scipy.stats import multivariate_normal
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans
import warnings
import networkx as nx
warnings.simplefilter(action='ignore', category=FutureWarning)
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
from scipy.linalg import sqrtm


def normalize_tracking_data(df):
    df[['xworld_norm', 'yworld_norm']] = df.groupby('frame_number')[['xworld', 'yworld']].transform(
        lambda x: x - x.mean()
    )
    return df

def initialize_roles_with_kmeans(df, n_roles): 
    kmeans = KMeans(n_clusters=len(n_roles), init='k-means++', random_state=42)
    normalized_positions = df[['xworld_norm', 'yworld_norm']].values
    df['specialized_role'] = kmeans.fit_predict(normalized_positions)
    initial_means = kmeans.cluster_centers_
    return df, initial_means


# PDF's stands for Probability Density Functions --> zu deutsch "Wahrscheinlichkeitsdichtefunktion"
def compute_cost_matrix(positions, role_pdfs, separation_bias=1.0): # separation_bias is a hyperparameter which influences the cost matrix, a higher value means a higher cost for players being close to each other
    n_positions = len(positions) # len of outer array, bc input is a 2D array
    n_roles = len(role_pdfs)
    cost_matrix = np.zeros((n_positions, n_roles)) # build the raw cost matrix

    for i, pos in enumerate(positions):
        for j, pdf in enumerate(role_pdfs):
            # Negative log-probability for cost, with a small separation bias
            cost_matrix[i, j] = -np.log(pdf.pdf(pos) + 1e-8) + separation_bias * np.sum(np.abs(pos - pdf.mean)) 
            # Manhattan distance
            # np.linalg.norm would be the euclidean distance
            # The cost is calculated as the negative log-probability of the position given the role, + a  bias term that encourages players to be spread out
            # it pemalizes points/players with low probability of being in a certain role and add the distance between the point and the role mean

    return cost_matrix


def update_roles(df, initial_means, separation_bias=6.0, cov_decay_factor=0.01):
    unique_roles = np.unique(df['specialized_role']) # array of unique roles - track_id [61, ..., 85]
    role_to_index = {role: idx for idx, role in enumerate(unique_roles)}


    # Initialize role PDFs with a large covariance in a dictionary
    role_pdfs = {
        role: multivariate_normal( # Multivariate normal distribution
            mean=initial_means[role_to_index[role]], cov=np.eye(2) * 20
        ) for role in unique_roles
    }
 
    colors = [plt.cm.tab10(i % 10) for i in range(len(unique_roles))]

    max_iterations = 20
    iteration = 0
    converged = False
    x = 0

    # Iteratively assign roles and refine PDFs
    while not converged and iteration < max_iterations:
        prev_assignments = df['specialized_role'].copy() # copy the specialized roles from the previous iteration
        new_role_positions = {role: [] for role in unique_roles} # creates a dictionary with empty lists for each role
        
        for _, frame_data in df.groupby('frame_number'): # interate over each frame-group
            positions = frame_data[['xworld_norm', 'yworld_norm']].values # 2D-array of positions [[x1, y1], [x2, y2], ...]
            cost_matrix = compute_cost_matrix(positions, [role_pdfs[role] for role in unique_roles], separation_bias) 
            row_ind, col_ind = linear_sum_assignment(cost_matrix) # Hungarian algorithm
            df.loc[frame_data.index, 'specialized_role'] = [unique_roles[j] for j in col_ind]  # assign potential new roles to the players

            for i, j in zip(row_ind, col_ind): # zip uses pairs of elements from the two lists
                new_role_positions[unique_roles[j]].append(positions[i]) # append the position to the corresponding role

        for role, positions in new_role_positions.items():
            if positions:
                positions = np.array(positions)
                mean = positions.mean(axis=0)
                cov_decay = max(cov_decay_factor, 10 / (iteration + 1))
                cov = np.cov(positions.T) + np.eye(2) * cov_decay
                role_pdfs[role] = multivariate_normal(mean=mean, cov=cov)

        iteration += 1
        converged = np.array_equal(prev_assignments.values, df['specialized_role'].values)
        print("Iteration:", iteration)

    #print("Final iteration:", iteration)
    return df, role_pdfs



def wasserstein_distance_gaussian(mean1, cov1, mean2, cov2):
    # Compute the squared difference between the means.
    mean_diff_sq = np.linalg.norm(mean1 - mean2)**2

    # Compute the matrix square root of cov1
    cov1_sqrt = sqrtm(cov1)
    
    # Compute the product cov1_sqrt * cov2 * cov1_sqrt
    product = cov1_sqrt @ cov2 @ cov1_sqrt
    
    # Compute the square root of the product matrix
    sqrt_product = sqrtm(product)
    
    # Compute the trace term
    trace_term = np.trace(cov1 + cov2 - 2 * sqrt_product)
    
    # Return the 2-Wasserstein distance (ensure non-negative)
    return np.sqrt(max(mean_diff_sq + trace_term, 0))


def compute_emd(formation1, formation2):
    n = len(formation1)
    m = len(formation2)
    cost_matrix = np.zeros((n, m))
    
    for i in range(n):
        for j in range(m):
            mean1, cov1 = formation1[i]
            mean2, cov2 = formation2[j]
            cost_matrix[i, j] = wasserstein_distance_gaussian(mean1, cov1, mean2, cov2)
    
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_ind, col_ind].sum()
    
    return total_cost

def extract_features(adj):
    G = nx.from_numpy_array(adj)
    features = []

    degrees = [d for _, d in G.degree()]
    features.append(np.mean(degrees))
    features.append(np.std(degrees))

    clustering = list(nx.clustering(G).values())
    features.append(np.mean(clustering))
    features.append(np.std(clustering))

    features.append(nx.density(G))

    betweenness = list(nx.betweenness_centrality(G).values())
    features.append(np.mean(betweenness))
    features.append(np.std(betweenness))

    

    return np.array(features)



def initialize_roles_with_kmeans_all(df, n_roles:int): 
    kmeans = KMeans(n_clusters=n_roles, init='k-means++', random_state=42)
    normalized_positions = df[['xworld_norm', 'yworld_norm']].values
    df['specialized_role'] = kmeans.fit_predict(normalized_positions)
    initial_means = kmeans.cluster_centers_
    return df, initial_means



def update_roles_all(df, initial_means, separation_bias=6.0, cov_decay_factor=0.01):
    unique_roles = np.unique(df['specialized_role']) # array of unique roles - track_id [61, ..., 85]
    role_to_index = {role: idx for idx, role in enumerate(unique_roles)}


    # Initialize role PDFs with a large covariance in a dictionary
    role_pdfs = {
        role: multivariate_normal( # Multivariate normal distribution
            mean=initial_means[role_to_index[role]], cov=np.eye(2) * 20
        ) for role in unique_roles
    }
 
    colors = [plt.cm.tab10(i % 10) for i in range(len(unique_roles))]

    max_iterations = 20
    iteration = 0
    converged = False
    x = 0

    # Iteratively assign roles and refine PDFs
    while not converged and iteration < max_iterations:
        prev_assignments = df['specialized_role'].copy() # copy the specialized roles from the previous iteration
        new_role_positions = {role: [] for role in unique_roles} # creates a dictionary with empty lists for each role
        
        for _, frame_data in df.groupby('frame'): # interate over each frame-group
            positions = frame_data[['xworld_norm', 'yworld_norm']].values # 2D-array of positions [[x1, y1], [x2, y2], ...]
            cost_matrix = compute_cost_matrix(positions, [role_pdfs[role] for role in unique_roles], separation_bias) 
            row_ind, col_ind = linear_sum_assignment(cost_matrix) # Hungarian algorithm
            df.loc[frame_data.index, 'specialized_role'] = [unique_roles[j] for j in col_ind]  # assign potential new roles to the players

            for i, j in zip(row_ind, col_ind): # zip uses pairs of elements from the two lists
                new_role_positions[unique_roles[j]].append(positions[i]) # append the position to the corresponding role

        for role, positions in new_role_positions.items():
            if positions:
                positions = np.array(positions)
                mean = positions.mean(axis=0)
                cov_decay = max(cov_decay_factor, 10 / (iteration + 1))
                cov = np.cov(positions.T) + np.eye(2) * cov_decay
                role_pdfs[role] = multivariate_normal(mean=mean, cov=cov)

        iteration += 1
        converged = np.array_equal(prev_assignments.values, df['specialized_role'].values)
        print("Iteration:", iteration)

    #print("Final iteration:", iteration)
    return df, role_pdfs