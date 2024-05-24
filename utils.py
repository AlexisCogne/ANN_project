import numpy as np
import torch
import gymnasium as gym
import A2C_continuous
import A2C_discrete
from tqdm import tqdm
import matplotlib.pyplot as plt

def set_seed(seed):
    torch.manual_seed(seed) # pytorch random seed
    np.random.seed(seed) # numpy random seed

def getTrajectory():
    # create a new sample environment to get new random parameters
    env = gym.make("InvertedPendulum-v4",max_episode_steps=1000) ## Continuous case
    states = []
    # get an initial state
    state, info = env.reset(seed=42)
    states.append(state)
    done = False
    # play one episode
    for i in range(6):
        # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
        state, reward, terminated, truncated, info = env.step([-0.2]) # only pushes left (0 is left, 1 is right)
        states.append(state)
        #pause for 0.5s
    while not done:
        # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
        state, reward, terminated, truncated, info = env.step([1]) # only pushes left (0 is left, 1 is right)
        states.append(state)
        done = terminated or truncated
    
    env.close()
    return states



def trainAgent(agents_seeds,n_seeds,envs,env_eval,n_updates,bool_discrete,obs_shape,action_space_dims,device = "cpu",critic_lr = 1e-3, actor_lr = 1e-5, n_envs = 1,n_steps_per_update = 1, evaluation_interval = 200000, n_eval_runs = 10,stochasticity_bool = True,stochastic_reward_probability = 0.9,gamma = 0.99):
    # LOGGED VARIABLES

    # per seed
    critic_losses = np.zeros((n_updates+1, n_seeds))
    actor_losses = np.zeros((n_updates+1, n_seeds))
    entropies = np.zeros((n_updates+1, n_seeds))
    values = [[] for _ in range(n_seeds)] # logs the values of the agent on the fixed trajectory
    evaluation_returns_seeds = [[] for _ in range(n_seeds)]

    # per worker (not used in the plots)
    episode_returns = [[] for _ in range(n_envs)] # logs the returns per episode per worker
    steps_episodes = [[] for _ in range(n_envs)] # logs the steps taken in each episode per worker

    # get the fixed trajectory for evaluation
    fixed_trajectory = getTrajectory()

    for s, agent_seed in enumerate(agents_seeds):
        print(f"Running seed {agent_seed} for agent {s}")
        if bool_discrete == True:
            agent = A2C_discrete.A2C(obs_shape, action_space_dims, device, critic_lr, actor_lr, n_envs)
        else:
            agent = A2C_continuous.A2C(obs_shape, action_space_dims, device, critic_lr, actor_lr, n_envs)

        # COUNTERS
        steps = 0
        steps_workers = [0] * n_envs
        ep_counter = 0
        ep_reward = [0] * n_envs

        # VARIABLE INITIALIZATION
        is_truncated = [False] * n_envs #creates a list of n_envs elements, all set to False
        is_terminated = [False] * n_envs #creates a list of n_envs elements, all set to False
        states = []

        set_seed(agent_seed)
        for i in range(n_envs):
            state, info = envs[i].reset(seed=agent_seed)  #only use the seed when resetting the first time
            states.append(state)


        # use tqdm to get a progress bar for training
        for steps in tqdm(range(n_updates+1)):
                

            # reset lists that collect experiences of a n_steps_per_update
            n_value_preds = torch.zeros(n_steps_per_update, n_envs, device=device)
            n_action_log_probs = torch.zeros(n_steps_per_update, n_envs, device=device)
            # don't take mask and reward gradient
            masks = torch.ones(n_steps_per_update, n_envs, device=device,requires_grad = False)
            n_rewards = torch.zeros(n_steps_per_update, n_envs, device=device,requires_grad = False)
            end_states = [[] for _ in range(n_envs)] # get a list for each env ex : [[], [], []]
            end_states_idx = [[0] for _ in range(n_envs)] # get a list for each env ex : [[0], [0], [0]]

            # play n_steps_per_update to collect data
            for step in range(n_steps_per_update):
                entropy = [0] * n_envs
                for env_idx in range(n_envs):
                    # select an action A_{t} using S_{t} as input for the agent, get action and values Vt
                    action, action_log_probs, V_t, entropy[env_idx] = agent.select_action(states[env_idx], bool_greedy=False)

                    # ensure no grad is taken in the step
                    with torch.no_grad():
                        # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
                        states[env_idx], reward, is_terminated[env_idx], is_truncated[env_idx], infos = envs[env_idx].step(
                            action.cpu().numpy()
                        )
                        steps_workers[env_idx] += 1
                    if stochasticity_bool:
                        # introduce stochasticity in the reward
                        if np.random.rand() < stochastic_reward_probability:
                            reward = 0

                    ep_reward[env_idx] += reward # increase episode return
                    mask = not is_terminated[env_idx] # define mask for bootstrapping

                    # log the value, reward and action log prob of the last step
                    n_value_preds[step][env_idx] = torch.squeeze(V_t)
                    n_rewards[step][env_idx] = torch.tensor(reward, device=device)
                    n_action_log_probs[step][env_idx] = action_log_probs

                    # for each env the mask is 1 if the episode is ongoing and 0 if it is terminated (not by truncation!)
                    masks[step][env_idx] = torch.tensor(mask) # allows for correct bootstrapping
                    
                    # reset environment if truncated or terminated
                    if is_terminated[env_idx] or is_truncated[env_idx]:
                        states_tensor = torch.tensor(states[env_idx], device=device) # Transform the last state reached (S_t+n) to a tensor
                        end_states[env_idx].append(states_tensor)
                        end_states_idx[env_idx].append(step)
                        states[env_idx], info = envs[env_idx].reset() # do not use the seed when resetting again
                        ep_counter += 1
                        steps_episodes[env_idx].append(steps_workers[env_idx])
                        episode_returns[env_idx].append(ep_reward[env_idx])
                        ep_reward[env_idx] = 0

            
            for env_idx in range(n_envs):
                # if statement to make sure we don't append the end state twice
                if not is_terminated[env_idx] and not is_truncated[env_idx]:
                    states_tensor = torch.tensor(states[env_idx], device=device) # Transform the last state reached (S_t+n) to a tensor
                    end_states[env_idx].append(states_tensor)
                    end_states_idx[env_idx].append(step)

            # calculate the losses for actor and critic
            critic_loss, actor_loss = agent.get_losses(
                n_rewards,
                n_action_log_probs,
                n_value_preds,
                masks,
                gamma,
                end_states,
                end_states_idx
            )

            # update the actor and critic networks
            agent.update_parameters(critic_loss, actor_loss)


            # log the losses and entropy
            critic_losses[steps, s] = critic_loss.detach().cpu().numpy()
            actor_losses[steps, s] = actor_loss.detach().cpu().numpy()
            entropies[steps, s] = sum(entropy) / len(entropy)


            #After every 20k steps, evaluate the performance of your agent by running it for 10 episodes with a greedy action policy (without noise)
            #on a newly initialized environment and plotting the evaluation statistics below.
            with torch.no_grad(): # No need to store gradients in the evaluation
                if steps % evaluation_interval == 0:
                    print("EVALUATION")
                    # evaluate
                    returns = []
                    episode_lengths = []
                    for i in range(n_eval_runs):
                        # Only use the seed when resetting at the beginning to ensure the states will follow the same initialization at every evalutaion step
                        if i == 0:
                            state, info = env_eval.reset(seed = agent_seed)
                            value = []
                            for state_traj in fixed_trajectory:
                                _, _, V_t, _ = agent.select_action(state_traj, bool_greedy=False)
                                value.append(V_t)
                            values[s].append(value)
                            
                        else:
                            state, info = env_eval.reset()
                        episode_return = 0
                        episode_length = 0
                        while True:
                            action, _, _, _ = agent.select_action(state,bool_greedy=True)
                            state, reward, terminated, truncated, info = env_eval.step(action.cpu().numpy())
                            episode_return += reward
                            episode_length += 1
                            if terminated or truncated:
                                break
                        returns.append(episode_return)
                        episode_lengths.append(episode_length)  
                    evaluation_returns_seeds[s].append(np.mean(returns))
    # Logging variables for each agent
    return(values,critic_losses,actor_losses,entropies,evaluation_returns_seeds)

# Aggregate function for plotting the 3 seeds together
def aggregate_plot(y1,y2,y3):
    """
    Aggregates three input curves by computing the element-wise minimum, maximum, and average.

    Parameters:
    y1, y2, y3: np.ndarray
        Input arrays representing the three curves to be aggregated. Each array should have the same shape.

    Returns:
    y_min, y_max, y_avg: np.ndarray
        The element-wise minimum / maximum / average of the three input curves.
    """

    # Compute minimum and maximum curves
    y_min = np.minimum(np.minimum(y1, y2), y3)
    y_max = np.maximum(np.maximum(y1, y2), y3)

    # Compute average curve
    y_avg = (y1 + y2 + y3) / 3

    return y_min, y_max, y_avg


def plotAggregated(values,n_seeds,agents_seeds,critic_losses,actor_losses,entropies,id_agent,n_steps_per_update,n_envs,n_steps,stochasticity_bool,evaluation_returns_seeds):

    rolling_length = 30 # Rolling length for the convolution

    # Arrays for the value function trajectories
    values_arr = np.array(values) # Transforming the list to a numpy array ==> Values_arr of size (n_seeds, n_eval_done, steps_in_trajectory, 1)
    values_sq = np.squeeze(values_arr) # Now of size (n_seeds, n_eval_done, steps_in_trajectory)
    n_eval_done = values_sq.shape[1] # Number of evaluations done 
    steps_in_trajectory = values_sq.shape[2] # Amount of steps in the trajectories of the value function
    n_traj = 4 # Amount of value function trajectories to be plotted

    # The code below is to select n_traj evenly spaced trajectories to plot between the first and last evaluation
    if n_eval_done >= n_traj:
        idx_traj = np.linspace(0, n_eval_done - 1, n_traj, dtype='int') # Selecting n_traj trajectories evenly spaced between the first and last
    else:
        idx_traj = np.arange(n_eval_done) # in case there are less evaluations than trajectories to plot, plot all evaluations

    n_traj = len(idx_traj) # No matter how many trajectories where found, n_traj is updated to the actual length of the idx_traj
    val_array = np.zeros((n_seeds, n_traj, steps_in_trajectory)) # Array to store the values of the selected trajectories
    val_array = values_sq[:,idx_traj] # of size n_seeds, n_traj, steps_in_trajectory => contains only the values of the selected trajectories

    traj_aggregates = np.zeros((n_seeds, n_traj, steps_in_trajectory)) # Will receive the values of y_min, y_max and y_avg for each of the n_traj for each seed


    # Creating the lists for the aggregation
    critic_y =[[] for _ in range(n_seeds)]
    actor_y =[[] for _ in range(n_seeds)]
    entropy_y =[[] for _ in range(n_seeds)]
    evaluation_returns_seeds = np.array(evaluation_returns_seeds)


    # Aggregating the losses and entropy while performing convolution
    for s, agent_seed in enumerate(agents_seeds):
        critic_y[s] = (
            np.convolve(np.array(critic_losses[:,s]), np.ones(rolling_length), mode="valid")
            / rolling_length)
        actor_y[s] = (
            np.convolve(np.array(actor_losses[:,s]), np.ones(rolling_length), mode="valid")
            / rolling_length)
        entropy_y[s] = (
            np.convolve(np.array(entropies[:,s]), np.ones(rolling_length), mode="valid")
            / rolling_length
        )
        
    # Building the y_min, y_max and y_avg for each of the plots
    critic_y_min, critic_y_max, critic_y_avg = aggregate_plot(critic_y[0],critic_y[1],critic_y[2])
    actor_y_min, actor_y_max, actor_y_avg = aggregate_plot(actor_y[0],actor_y[1],actor_y[2])
    entropy_y_min, entropy_y_max, entropy_y_avg = aggregate_plot(entropy_y[0],entropy_y[1],entropy_y[2])
    reward_y_min, reward_y_max, reward_y_avg = aggregate_plot(evaluation_returns_seeds[0], evaluation_returns_seeds[1], evaluation_returns_seeds[2])

    # x values for the plots
    critic_x = np.arange(0, critic_y_min.shape[0])
    actor_x = np.arange(0, actor_y_min.shape[0]) # Not necessary
    entropy_x = np.arange(0, entropy_y_min.shape[0]) # Not necessary
    reward_x = np.arange(0, reward_y_min.shape[0])
    traj_x = np.arange(0, steps_in_trajectory)
    #ep_return_x = np.arange(0, ep_return_y_min.shape[0])
    for j in range(n_traj): # Storing the y_min, y_max and y_avg of each of the n_traj trajectories
        traj_aggregates[0,j], traj_aggregates[1,j], traj_aggregates[2,j], = aggregate_plot(val_array[0][j], val_array[1][j], val_array[2][j])

    """ Plotting the losses, entropy and returns"""
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))
    fig.suptitle(f"Agent {id_agent} | Steps per update: {n_steps_per_update} | Workers: {n_envs} | Steps: {n_steps//1000}k | Stochasticity: {stochasticity_bool} | SEEDS: {agents_seeds}", x=0.5, fontsize = 12)

    # Critic loss
    axs[0, 0].fill_between(critic_x,critic_y_min, critic_y_max, color='gray', alpha=0.3, label='Min-Max Range')
    axs[0, 0].plot(critic_x, critic_y_avg, color='red', label='Average Curve')
    axs[0, 0].set_title('Critic Loss', fontweight='bold')
    axs[0, 0].set_yscale('log')  # Set log scale for the y-axis
    axs[0, 0].set_xlabel("Number of updates")

    # Actor loss
    axs[0, 1].fill_between(actor_x,actor_y_min, actor_y_max, color='gray', alpha=0.3, label='Min-Max Range')
    axs[0, 1].plot(actor_x, actor_y_avg, color='red', label='Average Curve')
    axs[0, 1].set_title('Actor Loss', fontweight='bold')
    axs[0, 1].set_yscale('log')  # Set log scale for the y-axis
    axs[0, 1].set_xlabel("Number of updates")

    # Entropy
    axs[1, 0].fill_between(entropy_x,entropy_y_min, entropy_y_max, color='gray', alpha=0.3, label='Min-Max Range')
    axs[1, 0].plot(entropy_x, entropy_y_avg, color='red', label='Average Curve')
    axs[1, 0].set_title("Entropy", fontweight='bold')
    axs[1, 0].set_xlabel("Number of updates")

    # Evaluation rewards
    axs[1, 1].fill_between(reward_x,reward_y_min, reward_y_max, color='gray', alpha=0.3, label='Min-Max Range')
    axs[1, 1].plot(reward_x, reward_y_avg, color='red', label='Average Curve')
    axs[1, 1].set_title('Evaluation Returns', fontweight='bold')
    axs[1, 1].set_xlabel("Evaluation rounds")

    #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(f'figures/Agent{id_agent}_Losses&Returns.png', bbox_inches='tight')
    plt.show()


    """Plotting the value function along the predefined fixed trajectory"""

    colors = ['blue', 'green', 'purple', 'orange', 'black', 'yellow', 'pink', 'brown', 'cyan', 'magenta']

    plt.figure(figsize=(10, 6))

    for j in range(len(idx_traj)):
        plt.fill_between(traj_x, traj_aggregates[0,j], traj_aggregates[1,j], color = colors[j], alpha=0.3, label='Min-Max Range')
        plt.plot(traj_x, traj_aggregates[2,j,:], color=colors[j], label=f"Evaluation {idx_traj[j]+1}")

    plt.suptitle(f"Agent {id_agent} | Steps per update: {n_steps_per_update} | Workers: {n_envs} | Steps: {n_steps//1000}k | Stochasticity: {stochasticity_bool} | SEEDS: {agents_seeds}", x=0.5, fontsize = 12)
    plt.title('Value Function on Fixed Trajectories', fontweight='bold')
    plt.xlabel('Agent step during the evaluation')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(f'figures/Agent{id_agent}_Trajectories.png', bbox_inches='tight')
    #plt.savefig(f'figures/Agent{id_agent}_Trajectories-&-{n_steps//1000}k_steps-&-{n_envs}_workers-&-{n_steps_per_update}_steps-per-update.png', bbox_inches='tight')
    plt.show()