import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict
from .sampler import NoiseScheduleVP, model_wrapper, DPM_Solver


class Swish(nn.Module):
    """Swish activation function"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(x) * x


class SimpleMLP(nn.Module):
    """
    Simple MLP diffusion model that imitates the Diffusion-BBO architecture.
    Takes input x, timestep t, and conditioning c (context + weights).
    
    Args:
        input_dim: Dimension of input x
        conditioning_dim: Dimension of conditioning vector c
        hidden_dim: Hidden layer dimension
        num_layers: Total number of layers (input + hidden + output)
        use_batch_norm: Whether to use batch normalization (default: False)
                       Batch normalization helps stabilize training by normalizing
                       activations. Note: BatchNorm1d uses running statistics during
                       inference, so it works even with batch_size=1.
    """

    def __init__(self, input_dim, conditioning_dim, hidden_dim=128, num_layers=4, use_batch_norm=False):
        super().__init__()

        self.input_dim = input_dim
        self.conditioning_dim = conditioning_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm

        # Build the MLP layers
        layers = []

        # Input layer: [input, timestep, conditioning]
        input_size = input_dim + 1 + conditioning_dim
        layers.append(nn.Linear(input_size, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(Swish())

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(Swish())

        # Output layer (no batch norm or activation for final layer)
        layers.append(nn.Linear(hidden_dim, input_dim))

        self.main = nn.Sequential(*layers)

    def forward(self, x, t, c):
        """
        Args:
            x: Input tensor [batch_size, input_dim]
            t: Timestep tensor [batch_size] or [batch_size, 1]
            c: Conditioning tensor [batch_size, conditioning_dim] (context + weights)
        """
        batch_size = x.shape[0]

        # Ensure t is the right shape
        if t.dim() == 1:
            t = t.unsqueeze(1).float()
        else:
            t = t.float()

        # Concatenate all inputs
        h = torch.cat([x, t, c], dim=1)

        # Forward pass
        noise_pred = self.main(h)

        return noise_pred


class SimpleDDPM(nn.Module):
    """
    Simple DDPM model for generating Pareto set solutions.
    Uses standard DDPM sampling instead of DDIM for simplicity.
    """

    def __init__(self, input_dim, conditioning_dim, timesteps=1000, hidden_dim=128, num_layers=4, use_batch_norm=False):
        super().__init__()

        self.input_dim = input_dim
        self.conditioning_dim = conditioning_dim
        self.timesteps = timesteps

        # Initialize the simple MLP model
        self.model = SimpleMLP(
            input_dim=input_dim,
            conditioning_dim=conditioning_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_batch_norm=use_batch_norm
        )

        # Initialize noise schedule (linear schedule like original DDPM)
        self.register_buffer('betas', self._linear_beta_schedule(timesteps))
        alphas = 1.0 - self.betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        self.register_buffer('alphas_cumprod_prev',
                             F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))

        # Precompute values for sampling
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1.0 - self.alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('posterior_variance',
                             self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))

    def _linear_beta_schedule(self, timesteps, beta_start=0.0001, beta_end=0.02):
        """Linear noise schedule"""
        return torch.linspace(beta_start, beta_end, timesteps)

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: add noise to x_start according to timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, c, noise=None):
        """
        Compute the loss for training.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t, c)

        loss = F.mse_loss(noise, predicted_noise)
        return loss

    def p_sample(self, x, t, c):
        """
        Single denoising step (reverse process).
        """
        betas_t = self.betas[t].reshape(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].reshape(-1, 1)

        # Predict noise
        predicted_noise = self.model(x, t, c)

        # Compute mean
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t].reshape(-1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def p_sample_loop(self, shape, c):
        """
        DDPM sampling loop.
        """
        device = next(self.parameters()).device

        # Start from random noise
        x = torch.randn(shape, device=device)

        for i in reversed(range(0, self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, c)
            x = torch.clamp(x, 0, 1)

        return x

    def sample(self, conditioning, num_samples):
        """
        Generate samples using DDPM sampling.

        Args:
            conditioning: Conditioning tensor [batch_size, conditioning_dim]
            num_samples: Number of samples to generate

        Returns:
            Generated samples [num_samples, input_dim]
        """
        if conditioning.dim() == 1:
            conditioning = conditioning.unsqueeze(0)

        if conditioning.shape[0] == 1 and num_samples > 1:
            conditioning = conditioning.repeat(num_samples, 1)

        shape = (num_samples, self.input_dim)

        with torch.no_grad():
            samples = self.p_sample_loop(shape, conditioning)

        # Ensure output shape matches expected shape
        assert samples.shape == (num_samples, self.input_dim), \
            f"Shape mismatch: expected ({num_samples}, {self.input_dim}), got {samples.shape}"

        # Clamp to [0, 1] range to handle boundary constraints
        # samples = torch.clamp(samples, 0, 1)

        return samples

    def sample_fast(self, conditioning, num_samples, steps=10, order=2, skip_type='time_uniform'):
        """
        Generate samples using DPM-Solver for faster sampling.

        Args:
            conditioning: Conditioning tensor [batch_size, conditioning_dim]
            num_samples: Number of samples to generate
            steps: Number of sampling steps (default: 10 for ~10x speedup)
            order: Order of DPM-Solver (1, 2, or 3, default: 2)
            skip_type: Type of time step spacing ('time_uniform', 'logSNR', or 'time_quadratic', default: 'time_uniform')

        Returns:
            Generated samples [num_samples, input_dim]
        """
        if conditioning.dim() == 1:
            conditioning = conditioning.unsqueeze(0)

        if conditioning.shape[0] == 1 and num_samples > 1:
            conditioning = conditioning.repeat(num_samples, 1)

        device = next(self.parameters()).device
        
        # Create noise schedule from existing betas/alphas_cumprod
        noise_schedule = NoiseScheduleVP(
            schedule='discrete',
            alphas_cumprod=self.alphas_cumprod,
            dtype=torch.float32
        )

        # Create a wrapper function for the model that handles conditioning
        # Since SimpleMLP can accept continuous t, we pass it directly
        # DPM-Solver expects model_fn(x, t_continuous) -> noise
        def dpm_model_fn(x, t_continuous):
            """
            Wrapper function for DPM-Solver that handles conditional generation.
            
            Args:
                x: Noisy input [batch_size, input_dim]
                t_continuous: Continuous time [batch_size] in [0, 1]
            
            Returns:
                Predicted noise [batch_size, input_dim]
            """
            # SimpleMLP can accept continuous t, so we pass it directly
            # The conditioning c is captured from the outer scope
            # Note: conditioning should already be expanded to match batch_size of x
            # (which should be num_samples, matching the shape of x_T we create below)
            batch_size = x.shape[0]
            
            # Ensure conditioning matches the batch size of x
            # This handles cases where DPM-Solver might process in smaller batches
            if conditioning.shape[0] != batch_size:
                # If batch sizes don't match, we need to handle this
                # For now, we'll use the first batch_size elements or repeat if needed
                if conditioning.shape[0] == 1:
                    # Single conditioning vector: repeat for all samples in batch
                    cond_batch = conditioning.repeat(batch_size, 1)
                elif conditioning.shape[0] >= batch_size:
                    # More conditionings than needed: use first batch_size
                    cond_batch = conditioning[:batch_size]
                else:
                    # Fewer conditionings: repeat the last one
                    cond_batch = torch.cat([
                        conditioning,
                        conditioning[-1:].repeat(batch_size - conditioning.shape[0], 1)
                    ], dim=0)
            else:
                cond_batch = conditioning
            
            return self.model(x, t_continuous, cond_batch)

        # Define correcting function to clamp x_t to [0, 1] at every step
        # This matches the behavior of the original DDPM sampling
        def correcting_xt_fn(xt, t, step):
            """
            Correct intermediate samples xt at each sampling step.
            Clamps values to [0, 1] range to match original DDPM behavior.
            
            Args:
                xt: Intermediate sample [batch_size, input_dim]
                t: Current time (not used but required by DPM-Solver)
                step: Current step (not used but required by DPM-Solver)
            
            Returns:
                Corrected xt clamped to [0, 1]
            """
            return torch.clamp(xt, 0, 1)

        # Create DPM-Solver instance
        dpm_solver = DPM_Solver(
            model_fn=dpm_model_fn,
            noise_schedule=noise_schedule,
            algorithm_type="dpmsolver++",
            correcting_x0_fn=None,
            correcting_xt_fn=correcting_xt_fn,  # Apply clamping at every step
        )

        # Generate samples
        shape = (num_samples, self.input_dim)
        x_T = torch.randn(shape, device=device)
        
        with torch.no_grad():
            # Sample from t=1.0 to t=1/timesteps (matching discrete-time DPM convention)
            t_start = 1.0
            t_end = 1.0 / self.timesteps
            
            samples = dpm_solver.sample(
                x=x_T,
                steps=steps,
                t_start=t_start,
                t_end=t_end,
                order=order,
                skip_type=skip_type,
                method='singlestep',
                lower_order_final=True,
                denoise_to_zero=False,
                solver_type='dpmsolver',
            )

        # Ensure output shape matches expected shape
        assert samples.shape == (num_samples, self.input_dim), \
            f"Shape mismatch: expected ({num_samples}, {self.input_dim}), got {samples.shape}"

        return samples

    def forward(self, x, c):
        """
        Forward pass for training.
        """
        batch_size = x.shape[0]
        device = x.device

        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()

        # Compute loss
        loss = self.p_losses(x, t, c)
        return loss


class SimpleParetoTrainer:
    """
    Simplified trainer class for Pareto set DDPM modeling.
    """

    def __init__(self,
                 input_dim,
                 conditioning_dim,
                 timesteps=1000,
                 hidden_dim=128,
                 num_layers=4,
                 learning_rate=1e-3,
                 batch_size=128,
                 epochs=50,
                 device=None,
                 trainer_id=None,
                 use_batch_norm=False):

        self.input_dim = input_dim
        self.conditioning_dim = conditioning_dim
        self.timesteps = timesteps
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create model
        self.model = SimpleDDPM(
            input_dim=input_dim,
            conditioning_dim=conditioning_dim,
            timesteps=timesteps,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_batch_norm=use_batch_norm
        ).to(self.device)

        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[int(self.epochs * 0.8), int(self.epochs * 0.9)],
            gamma=0.1
        )

        # Training logs
        self.logs = defaultdict(list)

    def prepare_data(self, X, contexts):
        """
        Prepare data for training.

        Args:
            X: Pareto set solutions [num_samples, input_dim]
            contexts: Combined context and weight vectors [num_samples, conditioning_dim]

        Returns:
            DataLoader for training
        """
        dataset_size = len(X)

        # Adjust batch size based on dataset size
        adjusted_batch_size = min(
            self.batch_size,
            max(1, dataset_size // 8)
        )

        dataset = TensorDataset(
            torch.FloatTensor(X),
            torch.FloatTensor(contexts)
        )

        return DataLoader(
            dataset=dataset,
            batch_size=adjusted_batch_size,
            shuffle=True
        )

    def train(self, X, contexts):
        """
        Train the DDPM model.

        Args:
            X: Pareto set solutions [num_samples, input_dim]
            contexts: Combined context and weight vectors [num_samples, conditioning_dim]

        Returns:
            Training logs
        """
        data_loader = self.prepare_data(X=X, contexts=contexts)

        for epoch in range(self.epochs):
            epoch_loss = 0
            num_batches = 0

            self.model.train()

            for iteration, batch in enumerate(data_loader):
                x, c = batch
                x, c = x.to(self.device), c.to(self.device)

                # Forward pass
                loss = self.model(x, c)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                # Print progress
                if iteration % max(1, len(data_loader) // 5) == 0:
                    print(f"Epoch {epoch + 1}/{self.epochs}, Batch {iteration + 1}/{len(data_loader)}, "
                          f"Loss: {loss.item():.4f}")

            self.scheduler.step()

            # Log epoch results
            avg_loss = epoch_loss / num_batches
            self.logs['loss'].append(avg_loss)

            print(f"Epoch {epoch + 1}/{self.epochs} completed, Avg Loss: {avg_loss:.4f}")

        return self.logs

    def generate_solutions(self, contexts, num_samples=10, use_fast_sampling=True, 
                          dpm_solver_steps=10, dpm_solver_order=2):
        """
        Generate new Pareto set solutions.

        Args:
            contexts: Combined context and weight vectors
            num_samples: Number of solutions to generate
            use_fast_sampling: If True, use DPM-Solver for faster sampling (default: True)
            dpm_solver_steps: Number of steps for DPM-Solver (default: 10)
            dpm_solver_order: Order of DPM-Solver (default: 2)

        Returns:
            Generated Pareto set solutions
        """
        self.model.eval()

        if not isinstance(contexts, torch.Tensor):
            contexts = torch.FloatTensor(contexts).to(self.device)

        # Expand contexts to match number of samples if needed
        if contexts.size(0) == 1 and num_samples > 1:
            contexts = contexts.repeat(num_samples, 1)

        if use_fast_sampling:
            generated_x = self.model.sample_fast(
                contexts, 
                num_samples, 
                steps=dpm_solver_steps,
                order=dpm_solver_order
            )
        else:
            generated_x = self.model.sample(contexts, num_samples)

        # Verify output shape
        expected_shape = (num_samples, self.model.input_dim)
        assert generated_x.shape == expected_shape, \
            f"Shape mismatch in generate_solutions: expected {expected_shape}, got {generated_x.shape}"

        return generated_x.cpu().numpy()

