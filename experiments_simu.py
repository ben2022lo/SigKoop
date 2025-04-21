import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sigkoop import SigKoop


dynamic_path = "simu/single_attractor.npy" #simu/single_attractor.npy / simu/duffing.npy
# load data
trajectories = np.load(dynamic_path)
X = torch.tensor(trajectories)

win = 10 # Windowsize
depth = 3  # Signature depth

train_idx, test_idx = train_test_split(list(range(X.shape[0])), train_size=0.8, random_state=42)

model = SigKoop(X, win, depth)
# with sliding window of size "win", extract paths from X and calculate signature features for each path
S_t, S_t_next = model.generate_signature_windows()
# train K
model.train_K(S_t[train_idx], S_t_next[train_idx])

# Train error
err_traj_train, err_time_train, train_error = model.compute_original_state_errors(S_t[train_idx], train_idx)
# Test error
err_traj_test, err_time_test, test_error = model.compute_original_state_errors(S_t[test_idx], test_idx)
model.error_traj_distributions(err_traj_train, err_traj_test)
model.error_time_distributions(err_time_train, err_time_test)
print(f"Train MSE: {train_error.item():.8f}")
print(f"Test  MSE: {test_error.item():.8f}")