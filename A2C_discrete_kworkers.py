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
        n_actions: int,
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
            nn.Linear(64, n_actions),
            nn.Softmax()
        ).to(self.device)
        
        #input is the state S(t), output is the value of the state V(s)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        ).to(self.device)

        # define optimizers for actor and critic using Adam optimizer
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)

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
        action_pd = self.actor(states)  # shape: [n_envs, n_actions]
        return (state_values, action_pd)
    

    def select_action(self, states: np.ndarray, bool_greedy)-> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # A COMMENTER

        state_values, action_pd = self.forward(states)
        action_pd = torch.distributions.Categorical(probs=action_pd) # A VERIFIER
        if bool_greedy:
            actions = action_pd.probs.argmax(dim=0)
        else:
            actions = action_pd.sample() # takes action with highest probability
        action_log_probs = action_pd.log_prob(actions) # purpose : measure of the log likelihood of the chosen actions under the current policy
        with torch.no_grad():
            entropy = action_pd.entropy()
        return (actions, action_log_probs, state_values, entropy)
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
        entropy: torch.Tensor,
        masks: torch.Tensor,
        gamma: float,
        #ent_coef: float, # Not used
        #device: torch.device, # Not used
        end_states: torch.Tensor, # VALUE OF LAST STATE POUR COMPUTE Q VALUES [n_envs, 4], 4 observations per state
        # Don't forget to call get_losses with states_tensor where: states_tensor = torch.tensor(states, device=device) 
        end_states_idx

    ):
        # A COMMENTER
     
        advantages_all = torch.zeros_like(rewards)
        for env_idx in range(self.n_envs):
           
            Ts = np.diff(end_states_idx[env_idx])
        
            #add +1 only to the first element of Ts
            Ts[0] = Ts[0] + 1
            advantages = torch.empty(0)

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

                    idx = end_states_idx[env_idx][i]
                    if i != 0:
                        idx+=1
                    Qvalues = torch.zeros(T,1)
                    Qval = self.critic(end_states[env_idx][i]) #get the value of the last state
                    Qval = torch.squeeze(Qval)
                    for t in reversed(range(T)): # t = T-1 to 0 !
                        if debugging:
                            print(f"t: {t}, idx: {idx}, end_states_idx[i+1]: {end_states_idx[env_idx][i+1]}")
                            
                        Qval = rewards[t+idx,env_idx] + masks[t+idx,env_idx] *gamma * Qval #idx starts at 0, we need to add it to shift the time index
                        Qvalues[t] = Qval
                        if debugging:
                            print(f"rewards[t+idx][env_idx]: {rewards[t+idx,env_idx]}, masks[t+idx]: {masks[t+idx]}, gamma: {gamma}, Qval: {Qval}")
                            print(f"Qvalues: {Qvalues}")
                 
                if debugging:       
                    print("Value preds: ", value_preds)
                    print("Small value preds: ", value_preds[idx:end_states_idx[env_idx][i+1]+1,env_idx])
                advantages_t = Qvalues - value_preds[idx:end_states_idx[env_idx][i+1]+1,env_idx].reshape(-1,1) #reshape to have the same shape as Qvalues
                advantages = torch.cat((advantages, advantages_t), 0)
            advantages_all[:,env_idx] = advantages.T





        # calculate the loss of the minibatch for actor and critic, it is a scalar tensor value : mean of all the advantages
        critic_loss = advantages.pow(2).mean() 
        actor_loss = -(advantages.detach() * action_log_probs).mean()

    
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
