import torch
import numpy as np
import signatory
import matplotlib.pyplot as plt


class SigKoop():
    def __init__(self, X, win, depth):
        super().__init__()
        self.X = X # full data set
        self.win = win # size of window / lenth of paths for signature calculation 
        self.N, self.T, self.D = X.shape
        self.num_windows = self.T - self.win
        
        self.depth = depth  # depth of signature features
        self.sig_dim = signatory.signature_channels(channels=self.D, depth=self.depth) # dimension of signature features
        self.K = torch.empty(self.sig_dim, self.sig_dim)
        
    def generate_signature_windows(self):
        """
        From a tensor X of shape (N, T, D), generate:
        - S_t: (N, T - t0, sig_dim)
        - S_t1: (N, T - t0, sig_dim)
        """
        
        S_t = torch.empty(self.N, self.num_windows, self.sig_dim, dtype=self.X.dtype, device=self.X.device)
        S_t_next = torch.empty(self.N, self.num_windows, self.sig_dim, dtype=self.X.dtype, device=self.X.device)

        for n in range(self.N):
            for t in range(self.num_windows):
                w1 = self.X[n:n+1, t:t + self.win, :]      # (1, self.win, self.D)
                w2 = self.X[n:n+1, t + 1:t + 1 + self.win, :]  # (1, self.win, self.D)
                S_t[n, t] = signatory.signature(w1, self.depth)
                S_t_next[n, t] = signatory.signature(w2, self.depth)

        return S_t, S_t_next

    def train_K(self, S_t_train, S_t_next_train):
        """
        Finds a finite Koopman operator K such that for all (n, t):
            S_t_train[n, t, :] @ K.T ≈ S_t_next_train[n, t, :]
        
        Args:
            S_t_train: Tensor of shape (self.N, self.T-self.win, self.sig_dim)
            S_t1_train: Tensor of shape (self.N, self.T-self.win, self.sig_dim)
        
        Returns:
            K: Tensor of shape (self.sig_dim, sefl.sig_dim)
        """

        # Flatten across all trajectories and timesteps: (N*T, sig_dim)
        Y = S_t_train.reshape(-1, self.sig_dim)
        Y_next = S_t_next_train.reshape(-1, self.sig_dim)
        

        # Solve least squares: K.T = (Y)^† Y_next
        # K = (Y.T @ Y)^-1 @ Y.T @ Y_next
        K_transpose, _, _, _ = np.linalg.lstsq(Y, Y_next, rcond=None)
        
        self.K = K_transpose.copy().T


    def compute_errors(self, S_t, S_t_next):
        """
        Computes prediction errors given by K_transpose.
        
        Returns:
            - error_per_traj: shape (M,)
            - error_per_time: shape (self.num_windows,)
            - overall_error: scalar
        """
        S_t_pred = S_t @ self.K.T  # shape (M, self.num_windows, self.sig_dim)
        mse = ((S_t_pred - S_t_next) ** 2).mean(dim=2)  # shape (M, self.num_windows)

        error_per_traj = mse.mean(dim=1)  # shape (M,)
        error_per_time = mse.mean(dim=0)  # shape (self.num_windows,)
        overall_error = mse.mean()        # scalar

        return error_per_traj, error_per_time, overall_error
    
    def compute_original_state_errors(self, S_t, idx):
        """
        Computes prediction errors on the original time series X,
        using only the first D components of signature prediction.

        Args:
            S_t: Tensor of shape (N, self.num_windows, self.sig_dim)
            idx: train_idx or test_idx

        Returns:
            - mse_per_traj: shape (M,)
            - mse_per_time: shape (self.num_windows,)
            - overall_mse: scalar
        """
        # Predict next signature: (M, self.num_windows, self.sig_dim)
        S_t_pred = S_t @ self.K.T

        # Use first D dims (increments) to predict X_{t+1}
        delta_pred = S_t_pred[:, :, :self.D]  # shape (M, self.num_windows, D)

        # Get X_t from original sequence
        X_first = self.X[idx, 1:-self.win+1, :]     # shape (M, self.num_windows, D), these are first elements of original paths of predicted signatures
        X_last = self.X[idx, self.win:, :] # shape (M, self.num_windows, D), these are last elements of original paths of predicted signatures

        # Predict X_{t+1} = X_t + Δ_pred
        X_pred = X_first + delta_pred
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))      
        for trajectory in X_pred:
            axs[0].plot(trajectory[:, 0], trajectory[:, 1], alpha=0.5)
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("x'")
        axs[0].set_title("Phase Portrait (prediction)")
        axs[0].grid()
        for trajectory in X_last:
            axs[1].plot(trajectory[:, 0], trajectory[:, 1], alpha=0.5)
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("x'")
        axs[1].set_title("Phase Portrait (true)")
        axs[1].grid()
        plt.show()

        # Compute MSE
        mse = ((X_pred - X_last) ** 2).mean(dim=2)  # (M, self.num_windows)
        
        mse_per_traj = mse.mean(dim=1)  # (M,)
        mse_per_time = mse.mean(dim=0)  # (self.num_windows,)
        overall_mse = mse.mean()        # scalar

        return mse_per_traj, mse_per_time, overall_mse
    def error_traj_distributions(self, err_traj_train, err_traj_test):
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        # Plot error per trajectory
        axs[0].hist(err_traj_train.cpu().numpy(), bins=30, color='skyblue', edgecolor='black')
        axs[0].set_title("Train error per Trajectory")
        axs[0].set_xlabel("MSE")
        axs[0].set_ylabel("Count")
        # Plot error per trajectory
        axs[1].hist(err_traj_test.cpu().numpy(), bins=30, color='skyblue', edgecolor='black')
        axs[1].set_title("Test error per Trajectory")
        axs[1].set_xlabel("MSE")
        axs[1].set_ylabel("Count")
        
        plt.tight_layout()
        plt.show()

    def error_time_distributions(self, err_time_train, err_time_test):
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Plot error per time step
        axs[0].plot(err_time_train.cpu().numpy(), marker='o', linestyle='-', color='orange')
        axs[0].set_title("Train error per Time Step")
        axs[0].set_xlabel("Time Step")
        axs[0].set_ylabel("MSE")

        # Plot error per time step
        axs[1].plot(err_time_test.cpu().numpy(), marker='o', linestyle='-', color='orange')
        axs[1].set_title("Test error per Time Step")
        axs[1].set_xlabel("Time Step")
        axs[1].set_ylabel("MSE")

        plt.tight_layout()
        plt.show()




