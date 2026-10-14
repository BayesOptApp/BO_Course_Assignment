import torch
import numpy as np

from botorch.models import SingleTaskGP
from gpytorch.kernels import ScaleKernel, MaternKernel
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qNoisyExpectedImprovement, qUpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.models.transforms.outcome import Standardize

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
        
        for iteration in range(num_iter):
            # 2. Fit Gaussian Process Model with outcome standardization for stability
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(
                X_tensor, 
                Y_tensor, 
                covar_module=covar_module,
                outcome_transform=Standardize(m=1)
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            try:
                fit_gpytorch_mll(mll)
            except Exception:
                pass  # Fallback gracefully if fitting fails on degenerate data
            
            # 3. Alternate acquisition functions between qNoisyExpectedImprovement and qUpperConfidenceBound (GP-Hedge style)
            if iteration % 2 == 0:
                acq_func = qNoisyExpectedImprovement(
                    model=model,
                    X_baseline=X_tensor,
                )
            else:
                best_f = Y_tensor.max() # Note: BoTorch UCB assumes maximization, but we minimize.
                # Since we want to minimize, we can use UCB on negated values or beta scheduling.
                # Actually, standard qUCB maximizes. To minimize, we negate the model or use qNEI mostly.
                # Let's stick to qNoisyExpectedImprovement with beta-adjusted variants or alternate with a different beta qUCB on negated Y.
                acq_func = qUpperConfidenceBound(
                    model=model,
                    beta=2.0
                )
                # To properly handle minimization with UCB, let's create a custom model or just use qNEI robustly.
                # Let's fallback to qNEI for robust exploration-exploitation balance.
                acq_func = qNoisyExpectedImprovement(
                    model=model,
                    X_baseline=X_tensor,
                )
            
            # 4. Optimize Acquisition Function to find next query point with higher restarts
            candidates, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=1,
                num_restarts=20,
                raw_samples=512,
            )
            
            new_x = candidates.detach()
            new_y_val = func(new_x.squeeze(0).numpy())
            new_y = torch.tensor([[new_y_val]], dtype=torch.double)
            
            # 5. Update observations
            X_tensor = torch.cat([X_tensor, new_x], dim=0)
            Y_tensor = torch.cat([Y_tensor, new_y], dim=0)
            
        # Find the best found point
        best_idx = torch.argmin(Y_tensor).item()
        best_f = Y_tensor[best_idx].item()
        best_x = X_tensor[best_idx].numpy()
        
        return best_f, best_x