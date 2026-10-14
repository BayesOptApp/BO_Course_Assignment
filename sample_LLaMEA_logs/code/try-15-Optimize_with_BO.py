import torch
import numpy as np

from botorch.models import SingleTaskGP
from gpytorch.kernels import ScaleKernel, MaternKernel
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qLogNoisyExpectedImprovement, qUpperConfidenceBound
from botorch.optim import optimize_acqf
from torch.quasirandom import SobolEngine

class Optimize_with_BO:
    def __init__(self, budget=50, dim=2, initial_points=10, random_seed=42):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        
        # Define bounds for the search space [-5, 5]^dim
        self.lb = -5.0
        self.ub = -5.0 + 10.0  # 5.0
        
    def __call__(self, func):
        # Set random seed for reproducibility
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # 1. Generate Initial Design of Experiments (DoE) using Sobol sequence
        sobol = SobolEngine(dimension=self.dim, scramble=True, seed=self.random_seed)
        initial_samples_scaled = sobol.draw(n=self.initial_points).to(dtype=torch.double)
        
        # Scale samples from [0, 1]^d to [lb, ub]^d
        X_tensor = initial_samples_scaled * (self.ub - self.lb) + self.lb
        
        # Evaluate initial points on the objective function
        Y_list = []
        for x in X_tensor:
            y_val = func(x.numpy())
            Y_list.append(y_val)
            
        Y_tensor = torch.tensor(Y_list, dtype=torch.double).unsqueeze(-1)
        
        # Define bounds tensor for BoTorch [2, dim]
        bounds = torch.tensor([[self.lb] * self.dim, [self.ub] * self.dim], dtype=torch.double)
        
        # Main BO Loop
        num_iter = max(0, self.budget - self.initial_points)
        
        # GP-Hedge acquisition function pool weights: qLogNEI, qUCB, and Kriging-Believer Thomson sampling proxy
        acq_weights = torch.ones(3, dtype=torch.double)
        
        for iteration in range(num_iter):
            # 2. Fit Gaussian Process Model with Matern 2.5 kernel and ard
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(X_tensor, Y_tensor, covar_module=covar_module)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            
            try:
                fit_gpytorch_mll(mll, options={"maxiter": 200, "disp": False})
            except Exception:
                pass  # Fallback gracefully if fitting encounters numerical issues
            
            # Dynamic beta for UCB scaling for deep multimodal exploration
            beta = 3.0 + 0.25 * iteration
            
            # 3. Define Acquisition Functions Portfolio
            acq_funcs = [
                qLogNoisyExpectedImprovement(model=model, X_baseline=X_tensor),
                qUpperConfidenceBound(model=model, beta=beta),
                # A secondary UCB with different exploration profile acting as Thompson sampling surrogate
                qUpperConfidenceBound(model=model, beta=max(0.5, beta * 0.5))
            ]
            
            # Choose acquisition function using softmax over hedge weights
            probs = torch.softmax(acq_weights, dim=0)
            chosen_idx = torch.multinomial(probs, 1).item()
            acq_func = acq_funcs[chosen_idx]
            
            # 4. Optimize Acquisition Function with high multi-starts to avoid local traps
            candidates, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=1,
                num_restarts=64,
                raw_samples=4096,
                options={"batch_limit": 5, "maxiter": 200}
            )
            
            new_x = candidates.detach()
            
            # Spatial repulsion/penalization check to prevent clustering around already explored local minima
            dists = torch.cdist(new_x, X_tensor)
            min_dist = dists.min().item()
            if min_dist < 0.03 * (self.ub - self.lb):
                # Orthogonal/random perturbation to escape local basins of attraction
                new_x = new_x + torch.randn_like(new_x) * 0.15 * (self.ub - self.lb)
                new_x = torch.clamp(new_x, self.lb, self.ub)
                
            new_y_val = func(new_x.squeeze(0).numpy())
            new_y = torch.tensor([[new_y_val]], dtype=torch.double)
            
            # Reward update for GP-Hedge (log-scaled improvement)
            current_best = Y_tensor.min().item()
            improvement = max(0.0, current_best - new_y_val)
            reward = np.log1p(improvement) * 10.0
            acq_weights[chosen_idx] = 0.8 * acq_weights[chosen_idx] + 0.2 * (reward + 0.01)
            
            # 5. Update observations
            X_tensor = torch.cat([X_tensor, new_x], dim=0)
            Y_tensor = torch.cat([Y_tensor, new_y], dim=0)
            
        # Find the best found point
        best_idx = torch.argmin(Y_tensor).item()
        best_f = Y_tensor[best_idx].item()
        best_x = X_tensor[best_idx].numpy()
        
        return best_f, best_x