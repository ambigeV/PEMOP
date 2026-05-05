import torch
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict


class ConditionalPrior(torch.nn.Module):
    """True-cVAE: Learnable conditional prior p(z|c)"""

    def __init__(self, context_size, latent_size, hidden_size=16):
        super().__init__()
        self.prior_network = torch.nn.Sequential(
            torch.nn.Linear(context_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 2 * latent_size)  # mean + log_var
        )
        self.latent_size = latent_size

    def forward(self, c):
        """True-cVAE: Get p(z|c) parameters"""
        params = self.prior_network(c)
        mean = params[:, :self.latent_size]
        log_var = torch.clamp(params[:, self.latent_size:], min=-10, max=10)
        return mean, log_var


class VAE(torch.nn.Module):
    """
    Variational Autoencoder for Pareto set/front modeling.
    Can be conditional on context variables.
    """

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=False, context_size=0, true_conditional=False):
        super().__init__()

        if conditional:
            assert context_size > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size
        self.conditional = conditional

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, context_size)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, context_size)

        # True-cVAE: Add conditional prior
        self.true_conditional = true_conditional and conditional
        if self.true_conditional:
            self.conditional_prior = ConditionalPrior(context_size, latent_size)
        else:
            self.conditional_prior = None

    def forward(self, x, c=None):
        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    # 7. Add sampling method to VAE (NEW METHOD)
    def sample_from_conditional_prior(self, c, num_samples=1):
        """True-cVAE: Sample from learned p(z|c)"""
        if self.true_conditional and c is not None:
            prior_mean, prior_log_var = self.conditional_prior(c)
            if num_samples > 1:
                prior_mean = prior_mean.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, self.latent_size)
                prior_log_var = prior_log_var.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, self.latent_size)

            std = torch.exp(0.5 * prior_log_var)
            eps = torch.randn_like(std)
            return prior_mean + eps * std
        else:
            # Fallback to standard prior
            batch_size = c.size(0) if c is not None else num_samples
            return torch.randn(batch_size, self.latent_size, device=next(self.parameters()).device)

    def inference(self, z, c=None):
        """
        Generate new Pareto set solutions from latent vectors.

        Args:
            z: Latent vectors
            c: Context vectors (if conditional)

        Returns:
            Generated Pareto set solutions
        """
        recon_x = self.decoder(z, c)
        return recon_x


class Encoder(torch.nn.Module):
    """
    Encoder network for VAE.
    Maps input (X) to latent distribution parameters.
    """

    def __init__(self, layer_sizes, latent_size, conditional, context_size):
        super().__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += context_size

        self.MLP = torch.nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name=f"L{i}", module=torch.nn.Linear(in_size, out_size))
            self.MLP.add_module(name=f"A{i}", module=torch.nn.ReLU())

        self.linear_means = torch.nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = torch.nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):
        if self.conditional and c is not None:
            x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(torch.nn.Module):
    """
    Decoder network for VAE.
    Maps latent vectors to reconstructed Pareto set solutions.
    """

    def __init__(self, layer_sizes, latent_size, conditional, context_size):
        super().__init__()

        self.MLP = torch.nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + context_size
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name=f"L{i}", module=torch.nn.Linear(in_size, out_size))
            if i + 1 < len(layer_sizes):
                self.MLP.add_module(name=f"A{i}", module=torch.nn.ReLU())
            else:
                pass
                # self.MLP.add_module(name="sigmoid", module=torch.nn.Sigmoid())
            # No activation in final layer - will be applied in forward method

    def forward(self, z, c=None):
        if self.conditional and c is not None:
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        # # Clamping output to [0,1] range
        x = torch.clamp(x, 0, 1)

        return x


class ParetoVAETrainer:
    """
    Trainer class for Pareto set/front VAE modeling.
    """

    def __init__(self,
                 input_dim,
                 output_dim=None,
                 latent_dim=2,
                 context_dim=0,
                 conditional=False,
                 true_conditional=False,
                 learning_rate=0.1,
                 batch_size=128,
                 epochs=100,
                 device=None,
                 trainer_id=None):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.conditional = conditional
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create model architecture
        # Default architecture with reasonable layer sizes
        encoder_sizes = [input_dim, max(input_dim, 2 * latent_dim, input_dim)]
        decoder_sizes = [max(input_dim, 2 * latent_dim, input_dim), input_dim]

        self.model = VAE(
            encoder_layer_sizes=encoder_sizes,
            latent_size=latent_dim,
            decoder_layer_sizes=decoder_sizes,
            conditional=conditional,
            context_size=context_dim
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # True-cVAE: Add prior optimizer if needed
        if true_conditional and conditional:
            self.true_conditional = True
            # Add conditional prior parameters to main optimizer
            self.optimizer = torch.optim.Adam(
                list(self.model.parameters()) + list(self.model.conditional_prior.parameters()),
                lr=learning_rate
            )
        else:
            self.true_conditional = False

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[int(self.epochs * 0.5), int(self.epochs * 0.75)],
            gamma=0.1
        )

        self.logs = defaultdict(list)
        self.logs['loss'] = []
        self.logs['mse'] = []
        self.logs['kld'] = []

    def loss_fn(self, recon_x, x, mean, log_var, c=None):
        """
        VAE loss function combining reconstruction loss and KL divergence.

        Args:
            recon_x: Reconstructed Pareto set
            x: Original Pareto set
            mean: Mean of latent distribution
            log_var: Log variance of latent distribution

        Returns:
            Combined loss
        """
        # MSE reconstruction loss
        MSE = torch.nn.functional.mse_loss(
            recon_x, x, reduction='sum')

        # True-cVAE: Use conditional KL divergence if enabled
        if self.true_conditional and c is not None:
            # KL(q(z|x,c) || p(z|c)) instead of KL(q(z|x,c) || p(z))
            prior_mean, prior_log_var = self.model.conditional_prior(c)
            posterior_var = torch.exp(log_var)
            prior_var = torch.exp(prior_log_var)

            KLD = 0.5 * torch.sum(
                prior_log_var - log_var +
                (posterior_var + (mean - prior_mean).pow(2)) / prior_var - 1
            )
            beta = 1.0

        else:
            # Original KL divergence
            KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            beta = 0.001


        # # KL divergence
        # KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        # beta = 0.001

        return (MSE + beta * KLD) / x.size(0), MSE / x.size(0), KLD / x.size(0)

    def prepare_data(self, X, Y=None, contexts=None):
        """
        Prepare data for training.

        Args:
            X: Pareto set solutions [batch_size, input_dim]
            Y: Pareto front values [batch_size, output_dim] - not used in current implementation
            contexts: Context/preference vectors [batch_size, context_dim]

        Returns:
            DataLoader for training
        """
        # Determine dataset size
        dataset_size = len(X)

        # Adjust batch size based on dataset size
        adjusted_batch_size = min(
            self.batch_size,  # Don't exceed configured max batch size
            max(1, dataset_size // 8)  # Aim for ~20 batches minimum
        )

        if contexts is not None and self.conditional:
            dataset = TensorDataset(
                torch.FloatTensor(X),
                torch.FloatTensor(contexts)
            )
        else:
            dataset = TensorDataset(torch.FloatTensor(X))

        # print(f"adjusted_batch_size is {adjusted_batch_size}")

        return DataLoader(
            dataset=dataset,
            batch_size=adjusted_batch_size,
            shuffle=True
        )

    def train(self, X, Y=None, contexts=None):
        """
        Train the VAE model.

        Args:
            X: Pareto set solutions [num_samples, input_dim]
            Y: Pareto front values [num_samples, output_dim] - not used in current implementation
            contexts: Context/preference vectors [num_samples, context_dim]

        Returns:
            Training logs
        """
        data_loader = self.prepare_data(X=X, contexts=contexts)

        for epoch in range(self.epochs):
            epoch_loss = 0
            epoch_mse = 0
            epoch_kld = 0

            for iteration, batch in enumerate(data_loader):
                if self.conditional:
                    x, c = batch
                    x, c = x.to(self.device), c.to(self.device)
                else:
                    x = batch[0].to(self.device)
                    c = None

                if self.conditional:
                    recon_x, mean, log_var, _ = self.model(x, c)
                else:
                    recon_x, mean, log_var, _ = self.model(x)

                loss, mse, kld = self.loss_fn(recon_x, x, mean, log_var, c)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()
                epoch_mse += mse.item()
                epoch_kld += kld.item()

                if (iteration + 1) % max(1, len(data_loader) // 5) == 0:
                    print(f"Epoch {epoch + 1}/{self.epochs}, Batch {iteration + 1}/{len(data_loader)}, "
                          f"Loss: {loss.item():.4f}")

            avg_loss = epoch_loss / len(data_loader)
            self.logs['loss'].append(avg_loss)
            self.logs['mse'].append(epoch_mse / len(data_loader))
            self.logs['kld'].append(epoch_kld / len(data_loader))
            print(f"Epoch {epoch + 1}/{self.epochs} completed, Avg Loss: {avg_loss:.4f}")

        return self.logs

    def validate(self, X_val, contexts_val=None):
        """Compute validation loss"""
        self.model.eval()
        with torch.no_grad():
            if self.conditional:
                recon_x, mean, log_var, _ = self.model(X_val, contexts_val)
            else:
                recon_x, mean, log_var, _ = self.model(X_val)

            loss, mse, kld = self.loss_fn(recon_x, X_val, mean, log_var)

        return loss.item(), mse.item(), kld.item()

    def generate_solutions(self, contexts=None, num_samples=10):
        """
        Generate new Pareto set solutions.

        Args:
            contexts: Context vectors to condition generation on
            num_samples: Number of solutions to generate per context

        Returns:
            Generated Pareto set solutions
        """
        self.model.eval()
        with torch.no_grad():
            if self.conditional and contexts is not None:
                # Convert contexts to tensor if needed
                if not isinstance(contexts, torch.Tensor):
                    contexts = torch.FloatTensor(contexts).to(self.device)

                # Expand contexts to match number of samples if needed
                if contexts.size(0) == 1 and num_samples > 1:
                    contexts = contexts.repeat(num_samples, 1)

                # num_contexts = contexts.size(0)
                # z = torch.randn(num_contexts, self.latent_dim).to(self.device)
                # True-cVAE: Sample from conditional prior instead of standard prior
                if self.true_conditional:
                    z = self.model.sample_from_conditional_prior(contexts, num_samples=1)
                else:
                    z = torch.randn(contexts.size(0), self.latent_dim).to(self.device)

                generated_x = self.model.inference(z, c=contexts)
            else:
                z = torch.randn(num_samples, self.latent_dim).to(self.device)
                generated_x = self.model.inference(z)

        return generated_x.cpu().numpy()




