import numpy as np
import torch
import gymnasium as gym
import A2C_continuous
import A2C_discrete
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def set_seed(seed):
    """
    Set the random seed for PyTorch and NumPy to ensure reproducibility.

    Parameters:
    - seed (int): The random seed value to set.

    Returns:
    - None
    """
    torch.manual_seed(seed) # pytorch random seed
    np.random.seed(seed) # numpy random seed

def getTrajectory(discrete_bool):
    """
    Generates arbitrary calibrated trajectories for an environment to make the pendulum fall after certain steps.

    Parameters:
    - discrete_bool (bool): A boolean indicating whether the environment is discrete or continuous.

    Returns:
    - states (list): A list containing the states of the environment over the trajectory.
    """
    # create a new sample environment to get new random parameters
    states = []
    done = False
    if discrete_bool: # Discrete case ==> Seed 40 and 5 steps left before pushing right
        env = gym.make("CartPole-v1", max_episode_steps=500)
        # get an initial state
        state, info = env.reset(seed=40)
        states.append(state)
        k = 0
        left, right = 0, 1

    else: # Continuous case ==> Seed 42 and 5 steps left [-0.2] before pushing right [0.5]
        env = gym.make("InvertedPendulum-v4", max_episode_steps=1000)
        # get an initial state
        state, info = env.reset(seed=42)
        states.append(state)
        k = 1
        left, right = [-0.2], [0.5]
    
    # play one episode
    for i in range(5+k):
        # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
        state, reward, terminated, truncated, info = env.step(left) # only pushes left (0 is left, 1 is right)
        states.append(state)
        #pause for 0.5s
    while not done:
        # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}

        state, reward, terminated, truncated, info = env.step(right) # only pushes right (0 is left, 1 is right)
        states.append(state)
        done = terminated or truncated
    env.close()
    return states

def trainAgent(n_steps, bootstrap, agents_seeds,n_seeds,envs,env_eval,n_updates,bool_discrete,obs_shape,action_space_dims,device = "cpu",critic_lr = 1e-3, actor_lr = 1e-5, n_envs = 1,n_steps_per_update = 1, evaluation_interval = 20000,logging_interval = 1000, n_eval_runs = 10,stochasticity_bool = True,stochastic_reward_probability = 0.9,gamma = 0.99):
    """
    Trains an agent using the Advantage Actor-Critic (A2C) algorithm.

    Parameters:
    - n_steps (int): Total number of training steps.
    - bootstrap (bool): Whether to use correct bootstrapping during training.
    - agents_seeds (list): List of random seeds for agents.
    - n_seeds (int): Number of agent seeds.
    - envs (list): List of training environments.
    - env_eval (environment): Evaluation environment.
    - n_updates (int): Number of updates to perform during training.
    - bool_discrete (bool): Whether the action space is discrete or continuous.
    - obs_shape (tuple): Shape of the observation space.
    - action_space_dims (int): Dimensionality of the action space.
    - device (str, optional): Device to use for computation (default is "cpu").
    - critic_lr (float, optional): Learning rate for the critic network (default is 1e-3).
    - actor_lr (float, optional): Learning rate for the actor network (default is 1e-5).
    - n_envs (int, optional): Number of parallel environments (default is 1).
    - n_steps_per_update (int, optional): Number of steps per update (default is 1).
    - evaluation_interval (int, optional): Interval for evaluating agent performance (default is 200000).
    - logging_interval (int, optional): Interval for logging training statistics (default is 1000).
    - n_eval_runs (int, optional): Number of evaluation runs (default is 10).
    - stochasticity_bool (bool, optional): Whether to introduce stochasticity in rewards (default is True).
    - stochastic_reward_probability (float, optional): Probability of introducing stochasticity in rewards (default is 0.9).
    - gamma (float, optional): Discount factor (default is 0.99).

    Returns:
    - tuple: Tuple containing logged variables including values, critic losses, actor losses, entropies, evaluation returns, training returns, and indices.
    """

    # per seed
    array_size = n_steps//1000+1 # Since we round down the values, when K > 1 or n > 1 the rounding prevents to log exactly the correct amount of values
    critic_losses = np.zeros((array_size, n_seeds))
    actor_losses = np.zeros((array_size, n_seeds))
    entropies = np.zeros((array_size, n_seeds))
    training_returns = []
    training_returns_idx = []
    values = [[] for _ in range(n_seeds)] # logs the values of the agent on the fixed trajectory
    evaluation_returns_seeds = [[] for _ in range(n_seeds)]

    # get the fixed trajectory for evaluation
    fixed_trajectory = getTrajectory(bool_discrete)

    for s, agent_seed in enumerate(agents_seeds):
        print(f"Running seed {agent_seed} for agent {s}")
        if bool_discrete == True:
            agent = A2C_discrete.A2C(obs_shape, action_space_dims, device, critic_lr, actor_lr, n_envs)
        else:
            agent = A2C_continuous.A2C(obs_shape, action_space_dims, device, critic_lr, actor_lr, n_envs)

        # COUNTERS
        # per worker and per seed (not returned or used)
        episode_returns = [[] for _ in range(n_envs)] # logs the returns per episode per worker
        steps_episodes = [[] for _ in range(n_envs)] # logs the steps taken in each episode per worker

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

        logging_counter = 1000 # Set to 1000 so we log the value at the beginning
        k_log = 0
        returns_log_bool = False
        # use tqdm to get a progress bar for training
        for steps in tqdm(range(n_updates+1)):
            logging_counter += 1

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
                    
                    ep_reward[env_idx] += reward # increase episode return before masking

                    if stochasticity_bool:
                        # introduce stochasticity in the reward
                        if np.random.rand() < stochastic_reward_probability:
                            reward = 0
                    if bootstrap:
                        mask = not is_terminated[env_idx] # define mask for bootstrapping
                    else:
                        mask = 1 # When the wrong bootstrapping needs to be run

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
            if logging_counter >= logging_interval: 
                logging_counter = 0
                
                # log the losses and entropy
                if k_log <= array_size-1: # We log the values until the array is full => We miss 1 value for K = 6 but it is not a big deal
                    critic_losses[k_log, s] = critic_loss.detach().cpu().numpy()
                    actor_losses[k_log, s] = actor_loss.detach().cpu().numpy()
                    entropies[k_log, s] = sum(entropy) / len(entropy)
                    k_log += 1

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

        steps_episodes_filtered,episode_returns_filtered = process_returns(steps_episodes,episode_returns)
        training_returns.append(episode_returns_filtered)
        training_returns_idx.append(steps_episodes_filtered)
    for i in range(n_seeds):
        if len(training_returns[i]) > array_size -1: #removing the last return in case it stops exactly at 500k
            training_returns[i].pop()
            training_returns_idx[i].pop()
        while len(training_returns[i]) < array_size -1:
            training_returns[i].append(training_returns[i][-1])
            training_returns_idx[i].append(training_returns_idx[i][-1])

    training_returns = np.array(training_returns).T # Transpose the array to have the correct shape
    training_returns_idx = np.array(training_returns_idx).T
        
    # Logging variables for each agent
    return(values,critic_losses,actor_losses,entropies,evaluation_returns_seeds,training_returns,training_returns_idx)

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

def aggregate_return_seeds(x):
    """
    Aggregate the returns of the training for each seed across episodes.

    Parameters:
    - x (numpy.ndarray): Array containing returns for each seed across episodes.

    Returns:
    - numpy.ndarray: Array containing aggregated returns for each seed.
    """
    x_array  = np.zeros(x.shape[0])

    for i in range(x.shape[0]):
        x_array[i] = (x[i, 0] + x[i, 1] + x[i, 2]) // 3 #Rounding the x_values to the nearest integer

    return x_array

def plotting(fig, axs, plot_traj, n_envs, n_steps_per_update, entropy_bool, compare_bool, critic_losses, evaluation_returns_seeds, values, agents_seeds, id_agent, color_agent, marker_style, linestyle, y_lim = [1e-5, 1e-1], n_col = [1,1], rolling_length = 1, entropies = None, loc = 0):
    """
    Plot the losses, entropy, and evaluation returns of the agent after training.

    Parameters:
    - fig (matplotlib.figure.Figure): The figure to which the subplots belong.
    - axs (np.ndarray): Array of subplots where the plots will be drawn.
    - n_envs (int): Number of parallel environments used for training.
    - n_steps_per_update (int): Number of steps per update during training.
    - plot_traj (bool): Boolean indicating whether to plot the trajectories.
    - entropy_bool (bool): Boolean indicating whether to plot entropy.
    - compare_bool (bool): Boolean indicating whether plotting is performed for comparison.
    - critic_losses (np.ndarray): Array containing critic losses during training.
    - evaluation_returns_seeds (list): List containing evaluation returns for different seeds.
    - values (list): List containing values of the agent on fixed trajectories.
    - agents_seeds (list): List of seeds used for training.
    - id_agent (int): Identifier for the agent.
    - color_agent (str): Color for the plots of the agent.
    - marker_style (str): Style of the marker for the plots.
    - linestyle (str): Style of the line for the plots.
    - y_lim (list, optional): Range for the y-axis. Default is [1e-5, 1e-1].
    - n_col (list, optional): Number of columns in the legends. Default is [1, 1].
    - rolling_length (int, optional): Rolling length for smoothing the plots. Default is 1.
    - entropies (np.ndarray, optional): Array containing entropies during training.
    - loc (int, optional): Location of the legend. Default is 0.
    """

    n_traj = 3
    n_seeds = len(agents_seeds) # Number of seeds used for training

    # Creating the list for the plots depending on which is specified
    if entropy_bool:
        entropy_y =[[] for _ in range(n_seeds)]
    else: 
        critic_y =[[] for _ in range(n_seeds)]
    
    for s, agent_seed in enumerate(agents_seeds):
        if entropy_bool:
            entropy_y[s] = (
                np.convolve(np.array(entropies[:,s]), np.ones(rolling_length), mode="valid")
                / rolling_length)
        else:
            critic_y[s] = (
            np.convolve(np.array(critic_losses[:,s]), np.ones(rolling_length), mode="valid")
            / rolling_length)
    
    # Building the y_min, y_max and y_avg for each of the plots
    if entropy_bool:
        entropy_y_min, entropy_y_max, entropy_y_avg = aggregate_plot(entropy_y[0],entropy_y[1],entropy_y[2])
    else:
        critic_y_min, critic_y_max, critic_y_avg = aggregate_plot(critic_y[0],critic_y[1],critic_y[2])
    evaluation_returns_seeds = np.array(evaluation_returns_seeds)
    reward_y_min, reward_y_max, reward_y_avg = aggregate_plot(evaluation_returns_seeds[0], evaluation_returns_seeds[1], evaluation_returns_seeds[2])

    # Arrays for the value function trajectories
    if plot_traj:
        values_arr = np.array(values) # Transforming the list to a numpy array ==> Values_arr of size (n_seeds, n_eval_done, steps_in_trajectory, 1)
        values_sq = np.squeeze(values_arr) # Now of size (n_seeds, n_eval_done, steps_in_trajectory)
        n_eval_done = values_sq.shape[1] # Number of evaluations done 
        steps_in_trajectory = values_sq.shape[2] # Amount of steps in the trajectories of the value function
        
        # The code below is to select n_traj evenly spaced trajectories to plot between the first and last evaluation
        if n_eval_done >= n_traj:
            idx_traj = np.linspace(0, n_eval_done - 1, n_traj, dtype='int') # Selecting n_traj trajectories evenly spaced between the first and last
            #idx_traj = np.array([0,1,steps_in_trajectory-2]) # For question 1 when trying to show wrong bootstrap ==> plotting the 2nd evaluation
        else:
            idx_traj = np.arange(n_eval_done) # in case there are less evaluations than trajectories to plot, plot all evaluations

        n_traj = len(idx_traj) # No matter how many trajectories where found, n_traj is updated to the actual length of the idx_traj
        val_array = np.zeros((n_seeds, n_traj, steps_in_trajectory)) # Array to store the values of the selected trajectories
        val_array = values_sq[:,idx_traj] # of size n_seeds, n_traj, steps_in_trajectory => contains only the values of the selected trajectories

        traj_aggregates = np.zeros((n_seeds, n_traj, steps_in_trajectory)) # Will receive the values of y_min, y_max and y_avg for each of the n_traj for each seed
        for j in range(n_traj): # Storing the y_min, y_max and y_avg of each of the n_traj trajectories
            traj_aggregates[0,j], traj_aggregates[1,j], traj_aggregates[2,j], = aggregate_plot(val_array[0][j], val_array[1][j], val_array[2][j])

    # Building the x axis for the plots
    if entropy_bool:
        x_axis = np.arange(0, entropy_y_min.shape[0]) * 1000
    else:
        x_axis = np.arange(0, critic_y_min.shape[0]) * 1000
    reward_x = np.arange(0, reward_y_min.shape[0])
    if plot_traj:
        traj_x = np.arange(0, steps_in_trajectory)

    # Plotting
    if entropy_bool:
        axs[0].fill_between(x_axis,entropy_y_min, entropy_y_max, color=color_agent[0], alpha=0.3, label=f'{id_agent} | Min-Max')
        axs[0].plot(x_axis, entropy_y_avg, color=color_agent[-1], label=f"{id_agent} | Average")
        axs[0].set_title("Entropy", fontweight='bold')
        axs[0].set_ylim(y_lim[0], y_lim[1])
        axs[0].set_xlabel("Number of steps")
        axs[0].legend(ncol = n_col[0])
    
    else:
        if compare_bool:
            axs[0].fill_between(x_axis,critic_y_min, critic_y_max, color=color_agent[0], alpha=0.3)
            axs[0].plot(x_axis, critic_y_avg, color=color_agent[-1], label=f"{id_agent} | K={n_envs} & n={n_steps_per_update}")
        else:
            axs[0].fill_between(x_axis,critic_y_min, critic_y_max, color=color_agent[0], alpha=0.3, label=f'{id_agent} | Min-Max')
            axs[0].plot(x_axis, critic_y_avg, color=color_agent[-1], label=f"{id_agent} | Average")
        axs[0].set_title('Critic Loss', fontweight='bold')
        axs[0].set_yscale('log')  # Set log scale for the y-axis
        axs[0].set_ylim(y_lim[0], y_lim[1])
        axs[0].set_xlabel("Number of steps")
        axs[0].legend(ncol = n_col[0])

    # Evaluation rewards
    if compare_bool:
        axs[1].fill_between(reward_x,reward_y_min, reward_y_max, color=color_agent[0], alpha=0.3)
        axs[1].plot(reward_x, reward_y_avg, color=color_agent[-1], linestyle = linestyle, label=f"{id_agent} | K={n_envs} & n={n_steps_per_update}")
    else:
        axs[1].fill_between(reward_x,reward_y_min, reward_y_max, color=color_agent[0], alpha=0.3, label=f'{id_agent} | Min-Max')
        axs[1].plot(reward_x, reward_y_avg, color=color_agent[-1], label=f"{id_agent} | Average")
    axs[1].set_title('Evaluation Returns', fontweight='bold')
    axs[1].set_xlabel("Evaluation rounds")
    axs[1].legend(ncol = n_col[0])

    # Trajectories
    if plot_traj: # Chooses whether to plot the trajectories or not
        for j in range(len(idx_traj)):
            if compare_bool:
                label = f'Eval {idx_traj[j]+1} | {id_agent}'
            else: # selects whether to add the min-max range or not
                label = f'Eval {idx_traj[j]+1} | Average'
                axs[2].fill_between(traj_x, traj_aggregates[0,j], traj_aggregates[1,j], color = color_agent[j], alpha=0.3, linestyle = linestyle, label=f'Eval {idx_traj[j]+1} | Min-Max')
            axs[2].plot(traj_x, traj_aggregates[2,j,:], color = color_agent[j], marker = marker_style, linestyle = linestyle, label=label)

        axs[2].set_title('Value Function on Fixed Trajectories', fontweight='bold')
        axs[2].set_xlabel('Agent step during the evaluation')
        axs[2].legend(ncol = n_col[1], loc = loc)
        axs[2].grid(False)

    return fig, axs

def time_plots(plt, agents, training_times, colors):
    """
    Plot the training times for each agent.

    Parameters:
    - plt: The matplotlib.pyplot module.
    - agents (list): List of agent names.
    - training_times (list): List of training times for each agent.
    - colors (list): List of colors for each agent's bar.

    Returns:
    - plt: The modified matplotlib.pyplot module.
    """
    bars = plt.bar(agents, training_times, color=colors)

    # Adding the values on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

    plt.title('Training Times for Each Agents', fontweight='bold')
    plt.ylabel('Training Time [s]')
    return plt

def lighten_color(color, factor=0.7):
    """
    Lighten the given color by a specified factor.

    Parameters:
    - color (str or tuple): Color to lighten in string (e.g., 'red') or tuple format (RGBA).
    - factor (float): Lightening factor, ranging from 0 to 1. Default is 0.7.

    Returns:
    - list: Lightened color in RGBA format.
    """
    base = mcolors.to_rgba(color)
    light = [1.0, 1.0, 1.0, 1.0]  # White color in RGBA
    result = [(1 - factor) * base[i] + factor * light[i] for i in range(4)]
    
    return result

# Code for the dictionnary storing the results from the training
def create_agent_data(agent_id, values, critic_losses, actor_losses, entropies, evaluation_returns_seeds, agents_seeds, n_steps, stochasticity_bool, n_envs, n_steps_per_update, training_returns_idx, training_returns, training_time):
    """
    Create a dictionary containing data for a specific agent.

    Parameters:
    - agent_id (str or int): Identifier for the agent.
    - values (list): List of values for the agent.
    - critic_losses (np.ndarray): Array containing critic losses during training.
    - actor_losses (np.ndarray): Array containing actor losses during training.
    - entropies (np.ndarray): Array containing entropies during training.
    - evaluation_returns_seeds (list): List containing evaluation returns for different seeds.
    - agents_seeds (list): List of seeds used for training.
    - n_steps (int): Number of training steps.
    - stochasticity_bool (bool): Boolean indicating whether stochasticity is enabled.
    - n_envs (int): Number of environments used for training.
    - n_steps_per_update (int): Number of steps per update during training.
    - training_returns_idx (list): List of training returns indices.
    - training_returns (list): List of training returns.
    - training_time (float): Training time in seconds.

    Returns:
    - dict: Dictionary containing agent data.
    """
    agent_data = {
        'values': values,
        'critic_losses': critic_losses,
        'actor_losses': actor_losses,
        'entropies': entropies,
        'evaluation_returns_seeds': evaluation_returns_seeds,
        'id': agent_id,
        'agents_seeds': agents_seeds,
        'n_steps': n_steps,
        'stochasticity_bool': stochasticity_bool,
        'n_envs': n_envs,
        'n_steps_per_update': n_steps_per_update,
        'train_returns_idx': training_returns_idx,
        'train_returns': training_returns,
        'training_time': training_time
    }
    return {agent_id: agent_data}

def save_agents_data(agents_data, filename):
    """
    Save the agents data dictionary as a .npz file.

    Parameters:
        agents_data (dict): The dictionary containing agent data.
        filename (str): The filename (including path if needed) to save the data.
    """
    np.savez(filename, **agents_data)

def load_agents_data(filename):
    """
    Load the agents data dictionary from a .npz file.

    Parameters:
        filename (str): The filename (including path if needed) from which to load the data.

    Returns:
        dict: The dictionary containing agent data.
    """
    loaded_data = np.load(filename, allow_pickle=True)
    reconstructed_agents_data = {}
    for key in loaded_data:
        reconstructed_agents_data[key] = loaded_data[key].item()
    return reconstructed_agents_data

# Function to get the first value and index per thousand group for a sublist
def get_first_per_thousand(sublist):
    """
    Get the first value encountered in each group of thousand values in the input sublist.

    Parameters:
    - sublist (list): List of values.

    Returns:
    - filtered_values (list): List containing the first value encountered in each group of thousand values.
    - indexes (list): List of tuples containing the index and value of the first value encountered in each group of thousand values.
    """
    thousands_encountered = set()
    filtered_values = []
    indexes = []
    for j, value in enumerate(sublist):
        if isinstance(value, int):
            thousand_group = value // 1000
            if thousand_group not in thousands_encountered:
                filtered_values.append(value)
                indexes.append((j, value))  # Store index and value as a tuple
                thousands_encountered.add(thousand_group)

    return filtered_values, indexes

def process_list_each1k(list_of_lists):
    """
    Process each sublist in the given list of lists to find the first value encountered in each group of thousand values.

    Parameters:
    - list_of_lists (list): List of sublists containing values.

    Returns:
    - final_filtered_values (list): List containing the smallest value encountered in each group of thousand values.
    - final_indexes (list): List of tuples containing the index and value of the smallest value encountered in each group of thousand values.
    """
    # Process each sublist
    results = []
    for i, sublist in enumerate(list_of_lists):
        _, index_value_pairs = get_first_per_thousand(sublist)
        results.append((i, index_value_pairs))

    # Combine the results and select the smallest value for each thousand group
    final_filtered_values = []
    final_indexes = []
    thousands_encountered = {}

    for i, index_value_pairs in results:
        for index, value in index_value_pairs:
            thousand_group = value // 1000
            if thousand_group not in thousands_encountered:
                thousands_encountered[thousand_group] = (i, index, value)
            else:
                # Compare and select the smallest value for the thousand group
                if value < thousands_encountered[thousand_group][2]:
                    thousands_encountered[thousand_group] = (i, index, value)

    # Extract the final filtered values and indexes
    for key in sorted(thousands_encountered.keys()):
        i, index, value = thousands_encountered[key]
        final_filtered_values.append(value)
        final_indexes.append((i, index))

    
    return final_filtered_values, final_indexes

def process_returns(train_returns_idx, train_returns):
    """
    Process the returns obtained during training to filter out only the returns obtained at the end of each thousand steps.

    Parameters:
    - train_returns_idx (list): List of lists containing the indices of returns obtained during training.
    - train_returns (list): List of lists containing the returns obtained during training.

    Returns:
    - train_returns_idx_filtered (list): List containing the indices of the returns obtained at the end of each thousand steps.
    - train_returns_filtered (list): List containing the returns obtained at the end of each thousand steps.
    """
    train_returns_idx_filtered, idx = process_list_each1k(train_returns_idx)
    # Extract values from the second list of lists using the final indexes
    train_returns_filtered = []
    for i, j in idx:
        train_returns_filtered.append(train_returns[i][j])

    return train_returns_idx_filtered,train_returns_filtered
