import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

class TokenizerMLP(nn.Module):
    def __init__(self, input_dim, output_dim=64, hidden_dims=[256, 128, 64]):
        super(TokenizerMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=64, n_heads=2, ff_dim=512, n_layers=4):
        super(TransformerEncoder, self).__init__()
        self.encoder_layers = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layers, num_layers=n_layers
        )

    def forward(self, tokens):
        return self.transformer_encoder(tokens)


class ActionHead(nn.Module):
    def __init__(self, input_dim=64, action_dim=32, hidden_dims=[1024, 512, 32]):
        super(ActionHead, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# -------------------------------
# TokenHSI Policy
# -------------------------------
class TokenHSIPolicy(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        state_dim=128,
        task_dims=[20, 38, 27, 42],
        action_dim=32,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=2,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self, clip_actions, clip_log_std, min_log_std, max_log_std
        )

        self.state_dim = state_dim
        self.task_dims = task_dims
        self.action_dim = action_dim
        self.num_tasks = len(task_dims)
        self.task_obs_size = sum(task_dims)
        # Proprioception tokenizer
        self.Tprop = TokenizerMLP(input_dim=state_dim)

        # Task tokenizers
        self.Ttask_list = nn.ModuleList(
            [TokenizerMLP(input_dim=dim) for dim in task_dims]
        )

        # Learnable embedding
        self.e = nn.Parameter(torch.randn(1, 1, 64))

        # Transformer Encoder
        self.transformer = TransformerEncoder(input_dim=64)

        # Action Head
        self.action_head = ActionHead(input_dim=64, action_dim=action_dim)

        # Fixed log standard deviation for policy
        self.log_std_parameter = nn.Parameter(
            torch.full((self.num_actions,), fill_value=-2.9), requires_grad=False
        )

    def compute(self, inputs, role):
        state = inputs["states"]  # Extract proprioception input

        #input["states"] contains proprioception and task observations and one-hot task assignment
        # we want to mask the task observations and one-hot task assignment for the inactive tasks
        # and concatenate the proprioception and active task observations
        proprioception = state[:, :self.state_dim]
        task_observations = state[:, self.state_dim:self.state_dim + sum(self.task_dims)]

        # Proprioception Token
        Tprop = self.Tprop(proprioception).unsqueeze(1)  # (B, 1, 64)

        # Use only proprioception and learnable embedding if no task info is present
        tokens = [Tprop, self.e.repeat(state.size(0), 1, 1)]

        # Task Tokens
        task_tokens = []
        start_idx = 0
        for t in range(self.num_tasks):
            task_obs = task_observations[:, start_idx:start_idx + self.task_dims[t]]
            task_tokens.append(self.Ttask_list[t](task_obs).unsqueeze(1))
            start_idx += self.task_dims[t]
        tokens = torch.cat(task_tokens, dim=1)  # (B, num_tasks, 64)


        # Transformer Encoder
        transformer_out = self.transformer(tokens)

        # Extract final token and get action
        final_token = transformer_out[:, -1, :]  # (B, 64)
        action_mean = self.action_head(final_token)  # (B, action_dim)

        return torch.tanh(action_mean), self.log_std_parameter, {}


# -------------------------------
# TokenHSI Value Function
# -------------------------------
class TokenHSIValueFunction(DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        state_dim=128,
        task_dims=[20, 38, 27, 42],
        hidden_dims=[2048, 1024, 512],
        clip_actions=False,
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        # Define dimensions
        self.state_dim = state_dim
        self.task_dims = task_dims
        self.task_obs_size = sum(task_dims)
        input_dim = state_dim + self.task_obs_size

        # Build MLP for value estimation
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def compute(self, inputs, role):
        state = inputs["states"][:, :self.state_dim + self.task_obs_size]
        # Pass through the value MLP
        value = self.network(state)  # (B, 1)
        return value, {}


# -------------------------------
# Conditional Discriminator
# -------------------------------
class TokenHSIConditionalDiscriminator(DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        hidden_dims=[1024, 512],
        clip_actions=False,
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)


        # Build MLP for discriminator
        layers = []
        prev_dim = self.num_observations
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def compute(self, inputs, role):
        state = inputs["states"]
        # Pass through the discriminator MLP
        validity = self.network(state)
        return validity, {}
