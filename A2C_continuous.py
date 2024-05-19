import numpy as np
import torch
import torch.nn as nn
from torch import optim

# THIS IMPLEMENTATION IS BASED ON THE FOLLOWING SITE : https://gymnasium.farama.org/tutorials/gymnasium_basics/vector_envs_tutorial/
class A2C(nn.Module):
    # A COMMENTER LINITIALISATION
    
    def __init__(
        self,
        state_dim: int,
        n_actions: int, ## No longer useful for Continuous
        device: torch.device,
        critic_lr: float = 1e-3,
        actor_lr: float = 1e-5,
        n_envs: int = 1,
    ):
        super().__init__()
        self.device = device
        self.n_envs = n_envs

        #input is the state S(t), output is the probability distribution over the actions
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            #nn.Softmax() ## Tell Riccardo I removed it 
        ).to(self.device)
        
        #input is the state S(t), output is the value of the state V(s)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        ).to(self.device)

        self.log_std = nn.Parameter(torch.zeros(self.n_envs, device=device), requires_grad = True)
        # define optimizers for actor and critic using Adam optimizer
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optim = optim.Adam(list(self.actor.parameters()) + [self.log_std], lr=actor_lr) ## Continuous case

    def forward(self, states: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        # UPDATE COMMENT A NOTRE SAUCE
        """
        Forward pass of the networks.
        Args:
            states: A batched vector of states.

        Returns:
            state_values: A tensor with the state values, with shape [n_envs,].
            action_logits_vec: A tensor with the action logits, with shape [n_envs, n_actions].
        """
        states = torch.Tensor(states).to(self.device)
        state_values = self.critic(states)  # shape: [n_envs,]
        action_mean = self.actor(states)  # shape: [n_envs, 1]
        exp_log_std = self.log_std.exp() # shape: [n_envs, 1] ## Verify with riccardo
        return (state_values, action_mean, exp_log_std)
    

    def select_action(self, states: np.ndarray, bool_greedy)-> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # A COMMENTER

        state_values, action_mean, exp_log_std = self.forward(states)
        action_distribution = torch.distributions.Normal(loc = action_mean, scale = exp_log_std) ## Continuous case: Normal distribution to sample the action (loc = µ, scale = σ)

        if bool_greedy:
            actions = action_mean
        else:
            actions = action_distribution.sample() # takes action with highest probability
        #action_log_probs = action_pd.log_prob(actions) # purpose : measure of the log likelihood of the chosen actions under the current policy
        #entropy = action_pd.entropy()

        clipped_actions = torch.clamp(actions, -3, 3) # clip the actions to be in the range [-1, 1]
        return (clipped_actions, state_values)
    """
    def get_losses(
        self,
        rewards: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        entropy: torch.Tensor,
        masks: torch.Tensor,
        gamma: float,
        ent_coef: float,
        device: torch.device,
    ):
    # A COMMENTER
        T = rewards.size(dim=0) # to get n_steps_per_update
        advantages = torch.zeros(T, self.n_envs, device=device)
        # compute the advantages
        #maks is 0 if the episode is done, 1 otherwise, we don't want to compute the advantage for the last step if its terminated
        for t in reversed(range(T - 1)):
            td_error = (
                rewards[t] + gamma * masks[t] * value_preds[t + 1] - value_preds[t]
            )
            advantages[t] = td_error
        # calculate the loss of the minibatch for actor and critic, it is a scalar tensor value : mean of all the advantages
        critic_loss = advantages.pow(2).mean() 

        # give a bonus for higher entropy to encourage exploration
        #actor_loss = ( -(advantages.detach() * action_log_probs).mean() - ent_coef * entropy.mean()) #.detach() is used to prevent the gradient from flowing back into the actor network
        
        #sans entropy bonus
        actor_loss = -(advantages.detach() * action_log_probs).mean()
        return (critic_loss, actor_loss)
    """
    
    def get_losses(
        self,
        rewards: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        #entropy: torch.Tensor, ## For continuous case
        masks: torch.Tensor,
        gamma: float,
        #ent_coef: float, # Not used
        #device: torch.device, # Not used
        end_states: torch.Tensor, # VALUE OF LAST STATE POUR COMPUTE Q VALUES [n_envs, 4], 4 observations per state
        # Don't forget to call get_losses with states_tensor where: states_tensor = torch.tensor(states, device=device) 
        end_states_idx: list[int]

        ):
        # Ensure all tensors are of the same dtype (torch.float32)
        # rewards = rewards.to(dtype=torch.float32)
        # action_log_probs = action_log_probs.to(dtype=torch.float32)
        # value_preds = value_preds.to(dtype=torch.float32)
        masks = masks.to(dtype=torch.float32)
        for i in range(len(end_states)):
            end_states[i] = end_states[i].to(dtype=torch.float32)
    # A COMMENTER
        # FOR 1 WORKER
        Ts = np.diff(end_states_idx)
    
        #add +1 only to the first element of Ts
        Ts[0] = Ts[0] + 1
        advantages = torch.empty(0, dtype=torch.float32, device=self.device)

        if 0 in masks:
            debugging = False
            if debugging:
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                print("DEBUGGING")
                print(f"end_states: {end_states}")
                print(f"end_states_idx: {end_states_idx}")
                print(f"mask: {masks}")
                print(f"Ts: {Ts}")
        else:
            debugging = False
        

        for i, T in enumerate(Ts):
            if debugging:
                print(f"i: {i}, T: {T}")
            with torch.no_grad():

                idx = end_states_idx[i]
                if i != 0:
                    idx+=1
                Qvalues = torch.zeros(T, 1, dtype=torch.float32, device=self.device)
                print( end_states[i])
                Qval = self.critic(end_states[i]) #get the value of the last state
                #Qval = torch.squeeze(Qval).to(dtype=torch.float32)
                for t in reversed(range(T)): # t = T-1 to 0 !
                    if debugging:
                        print(f"t: {t}, idx: {idx}, end_states_idx[i+1]: {end_states_idx[i+1]}")
                        
                    Qval = rewards[t+idx] + masks[t+idx] *gamma * Qval #idx starts at 0, we need to add it to shift the time index
                    Qvalues[t] = Qval
                    if debugging:
                        print(f"rewards[t+idx+1]: {rewards[t+idx]}, masks[t+idx+1]: {masks[t+idx]}, gamma: {gamma}, Qval: {Qval}")
                        print(f"Qvalues: {Qvalues}")
            if debugging:       
                print("Value preds: ", value_preds)
                print("Small value preds: ", value_preds[idx:end_states_idx[i+1]+1])
            advantages_t = Qvalues - value_preds[idx:end_states_idx[i+1]+1]
            advantages = torch.cat((advantages, advantages_t), 0)
        




        # calculate the loss of the minibatch for actor and critic, it is a scalar tensor value : mean of all the advantages
        critic_loss = advantages.pow(2).mean() 
        actor_loss = -(advantages.detach() * action_log_probs).mean()
        print("ACtion log probs", action_log_probs.requires_grad)
        print("Actor loss post: ", actor_loss.requires_grad)

    
        return (critic_loss, actor_loss)
    
    def update_parameters(self, critic_loss: torch.Tensor, actor_loss: torch.Tensor):
        """
        Updates the parameters of the actor and critic networks.

        Args:
            critic_loss: The critic loss.
            actor_loss: The actor loss.
        """
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
