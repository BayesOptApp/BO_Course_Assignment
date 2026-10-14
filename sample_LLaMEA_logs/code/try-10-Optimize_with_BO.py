import torch
import numpy as np

from botorch.models import SingleTaskGP
from gpytorch.kernels import ScaleKernel, MaternKernel
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qLogNoisyExpectedImprovement, qUpperConfidenceBound, qExpectedImprovement
from botorch.optim import optimize_acqf

class Optimize_with_BO:
    def __init__(self, budget=50, dim=2, initial_points=10, random_seed=42):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        
        # Define bounds for the search space [-5, 5]^dim
        self.lb = -5.0
        self.ub = 5.0
        
    def __call__(self, func):
        from scipy.stats import qmc

        # Set random seed for reproducibility
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # 1. Generate Initial Design of Experiments (DoE) using Latin Hypercube Sampling (LHS)
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.random_seed)
        initial_samples_scaled = sampler.random(n=self.initial_points)
        
        # Scale samples from [0, 1]^d to [lb, ub]^d
        X_init = initial_samples_scaled * (self.ub - self.lb) + self.lb
        X_tensor = torch.tensor(X_init, dtype=torch.double)
        
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
        
        # GP-Hedge acquisition function pool setup with three complementary acquisition functions
        acq_weights = torch.ones(3, dtype=torch.double)
        
        for iteration in range(num_iter):
            # 2. Fit Gaussian Process Model with jitter/noise handling
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(X_tensor, Y_tensor, covar_module=covar_module)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            
            try:
                fit_gpytorch_mll(mll)
            except Exception:
                pass  # Fallback gracefully if fitting encounters numerical issues
            
            # Dynamic beta for UCB that scales with iterations to balance exploration/exploitation
            beta = 2.5 + 0.05 * iteration
            
            # 3. Define Acquisition Functions for GP-Hedge portfolio
            acq_funcs = [
                qLogNoisyExpectedImprovement(model=model, X_baseline=X_tensor),
                qUpperConfidenceBound(model=model, beta=beta),
                qExpectedImprovement(model=model, best_f=Y_tensor.min().item())
            ]
            
            # Choose acquisition function using a softmax over hedge weights (GP-Hedge strategy)
            probs = torch.softmax(acq_weights, dim=0)
            chosen_idx = torch.multinomial(probs, 1).item()
            acq_func = acq_funcs[chosen_idx]
            
            # 4. Optimize Acquisition Function to find next query point with robust multi-starts
            candidates, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=1,
                num_restarts=40,
                raw_samples=2048,
            )
            
            new_x = candidates.detach()
            new_y_val = func(new_x.squeeze(0).numpy())
            new_y = torch.tensor([[new_y_val]], dtype=torch.double)
            
            # Reward update for GP-Hedge (improvement over current best)
            current_best = Y_tensor.min().item()
            reward = max(0.0, current_best - new_y_val)
            acq_weights[chosen_idx] += reward
            
            # 5. Update observations
            X_tensor = torch.cat([X_tensor, new_x], dim=0)
            Y_tensor = torch.cat([Y_tensor, new_y], dim=0)
            
        # Find the best found point
        best_idx = torch.argmin(Y_tensor).item()
        best_f = Y_tensor[best_idx].item()
        best_x = X_tensor[best_idx].numpy()
        
        return best_f, best_x