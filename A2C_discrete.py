import numpy as np
import torch
import torch.nn as nn
from torch import optim

class A2C(nn.Module):
    """
    A2C (Advantage Actor-Critic) class for reinforcement learning DISCRETE CASE.

    Args:
        state_dim (int): The dimension of the input state for the actor and critic networks.
        n_actions (int): The number of actions for the actor network.
        device (torch.device): The device to run the networks on.
        critic_lr (float, optional): The learning rate for the critic network. Defaults to 1e-3.
        actor_lr (float, optional): The learning rate for the actor network. Defaults to 1e-5.
        n_envs (int, optional): The number of environments. Defaults to 1.
    """

    
    def __init__(
        self,
        state_dim: int, # for the input layer of the actor and critic
        n_actions: int, # for the output layer of the actor
        device: torch.device,
        critic_lr: float = 1e-3,
        actor_lr: float = 1e-5,
        n_envs: int = 1,
    ):
        super().__init__()
        self.device = device
        self.n_envs = n_envs

        # Input is the state S(t), output is the probability distribution over the actions.
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions),
            nn.Softmax()
        ).to(self.device)
        
        # Input is the state S(t), output is the value of the state V(s).
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        ).to(self.device)

        # Define optimizers for actor and critic using Adam optimizer.
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)

    def forward(self, state: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the networks.

        Args:
            state (np.ndarray): A state. EX : [-0.10563138 -0.6316567   0.1910039   1.1486845 ]

        Returns: 
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the state values and action logits.
                - state_values (torch.Tensor): A tensor with the state values, with shape [1]. EX : tensor([0.0473])
                - action_logits_vec (torch.Tensor): A tensor with the action logits, with shape [1,n_actions]. EX : tensor([0.5145, 0.4855])
        """
        state = torch.Tensor(state).to(self.device)
        state_values = self.critic(state)  # shape: [1]
        action_pd = self.actor(state)  # shape: [1, n_actions]
        return (state_values, action_pd)

    def select_action(self, state: np.ndarray, bool_greedy) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Selects an action based on the current state.

        Args:
            state (np.ndarray): A state. EX : [-0.10563138 -0.6316567   0.1910039   1.1486845 ]
            bool_greedy (bool): A flag indicating whether to select the action greedily or stochastically.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the selected action, action log probabilities, state value, and entropy.
        """ 
        state_value, action_pd = self.forward(state)
        action_pd = torch.distributions.Categorical(probs=action_pd)
        if bool_greedy:
            action = action_pd.probs.argmax(dim=0)
        else:
            action = action_pd.sample() # sample an action from the distribution
        action_log_probs = action_pd.log_prob(action) # purpose : measure of the log likelihood of the chosen actions under the current policy
        with torch.no_grad():
            entropy = action_pd.entropy()
        return (action, action_log_probs, state_value, entropy)

    def get_losses(
        self,
        rewards: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        masks: torch.Tensor,
        gamma: float,
        end_states: torch.Tensor,
        end_states_idx
    ):
        """
        Calculates the losses for the actor and critic networks.

        Args:
            rewards (torch.Tensor): The rewards received at each step. Shape : [n_steps_per_update, n_envs]
            action_log_probs (torch.Tensor): The log probabilities of the selected actions at each step. Shape : [n_steps_per_update, n_envs]
            value_preds (torch.Tensor): The predicted values of the states at each step. Shape : [n_steps_per_update, n_envs]
            masks (torch.Tensor): The masks indicating whether the episode is done or not at each step. Shape : [n_steps_per_update, n_envs]
            gamma (float): The discount factor. 
            end_states (list of lists of states): The values of the last states for computing Q-values. length : n_envs. EX : [[tensor([ 0.1559,  1.3362, -0.2230, -2.2074])], [tensor([-0.1275, -1.2164,  0.2200,  2.0348]), tensor([-0.0017,  0.4350,  0.0262, -0.5808])]] (2 envs)
            end_states_idx (list of lists of indexes): The indices of the end states. length : n_envs. EX :[[0, 8], [0, 6, 8]] (2 envs)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the critic loss and actor loss.
        """
        advantages_all = torch.zeros_like(rewards) # shape: [n_steps_per_update, n_envs]

        # Loop on the environments to compute the advantages
        for env_idx in range(self.n_envs): 

            # find all the length of the episodes for correct bootstrapping
            Ts = np.diff(end_states_idx[env_idx])
            Ts[0] = Ts[0] + 1 # add +1 only to the first element of Ts for correct T value

            advantages = torch.empty(0)
            
            # Loop over the episodes to compute the advantages
            for i, T in enumerate(Ts):
                with torch.no_grad(): 
                    idx = end_states_idx[env_idx][i]
                    if i != 0:
                        idx += 1 # to avoid the overlap between the episodes

                    Qvalues = torch.zeros(T, 1)
                    Qval = self.critic(end_states[env_idx][i]) # compute the V-value of the last state of the episode (called Qval for simplicity in the following algorithm)
                    Qval = torch.squeeze(Qval) 

                    # Loop over the steps of the episode to bootstrap the Q-values
                    for t in reversed(range(T)): # t = T-1 to 0 !
                        Qval = rewards[t + idx, env_idx] + masks[t + idx, env_idx] * gamma * Qval
                        Qvalues[t] = Qval

                # compute the advantages over the episode and concatenate them
                advantages_t = Qvalues - value_preds[idx:end_states_idx[env_idx][i+1]+1, env_idx].reshape(-1, 1)
                advantages = torch.cat((advantages, advantages_t), 0)
            
            #concatenate the advantages of all the steps of all the workers
            advantages_all[:, env_idx] = advantages.T

        # Compute the critic loss and the actor loss
        critic_loss = advantages.pow(2).mean()
        actor_loss = -(advantages.detach() * action_log_probs).mean()

        return (critic_loss, actor_loss)

    def update_parameters(self, critic_loss: torch.Tensor, actor_loss: torch.Tensor):
        """
        Updates the parameters of the actor and critic networks.

        Args:
            critic_loss (torch.Tensor): The critic loss.
            actor_loss (torch.Tensor): The actor loss.
        """
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
