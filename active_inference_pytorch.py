import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt

# --- Environment ---

class GridWorld:
    """
    A simple 10x10 grid world POMDP environment for the active inference agent.

    The agent's state is its (x, y) coordinate. The agent receives a noisy
    observation of its state. The goal is to navigate to a fixed target location.
    The agent can take pragmatic actions (move) or an epistemic action (get a hint),
    which reduces the noise of its observations.
    """
    def __init__(self, size=10):
        """
        Initializes the grid world environment.
        Args:
            size (int): The size of one side of the square grid.
        """
        self.size = size
        self.goal_location = np.array([size - 1, size - 1])
        self.agent_location = None
        self.noise_level = None
        self.action_space_size = 5 # 0:up, 1:down, 2:left, 3:right, 4:hint
        self.reset()

    def reset(self):
        """
        Resets the environment to a new episode.
        The agent is placed at a random starting location.
        Returns:
            tuple: A tuple containing the initial noisy observation and the true starting state.
        """
        self.agent_location = np.random.randint(0, self.size, size=2)
        self.noise_level = 1.0
        return self._get_observation(), self.agent_location.copy()

    def step(self, action):
        """
        Executes one time step in the environment.
        Args:
            action (int): The action selected by the agent.
        Returns:
            tuple: A tuple containing the new noisy observation and the true new state.
        """
        if action == 0:  # Up
            self.agent_location[0] = max(0, self.agent_location[0] - 1)
        elif action == 1:  # Down
            self.agent_location[0] = min(self.size - 1, self.agent_location[0] + 1)
        elif action == 2:  # Left
            self.agent_location[1] = max(0, self.agent_location[1] - 1)
        elif action == 3:  # Right
            self.agent_location[1] = min(self.size - 1, self.agent_location[1] + 1)
        elif action == 4:  # Hint (epistemic action)
            self.noise_level = max(0.1, self.noise_level * 0.5)

        return self._get_observation(), self.agent_location.copy()

    def _get_observation(self):
        """
        Generates a noisy observation of the agent's current location.
        Returns:
            np.ndarray: The noisy observation of the agent's coordinates.
        """
        noise = np.random.normal(0, self.noise_level, size=2)
        observation = self.agent_location + noise
        return np.clip(observation, 0, self.size - 1)

# --- Agent Components ---

class PosteriorNet(nn.Module):
    """
    A neural network that approximates the posterior distribution over hidden states.
    It infers the hidden state (true coordinates) from a noisy observation and the previous action.
    This corresponds to minimizing the complexity term of the Variational Free Energy (VFE).
    """
    def __init__(self, input_size=7, hidden_size=64, output_size=2):
        """
        Initializes the posterior network.
        Args:
            input_size (int): Size of the input (observation + one-hot action).
            hidden_size (int): Size of the hidden layers.
            output_size (int): Size of the output (inferred state).
        """
        super(PosteriorNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, observation, last_action):
        """
        Performs a forward pass through the network.
        Args:
            observation (torch.Tensor): The noisy observation.
            last_action (torch.Tensor): The last action taken by the agent.
        Returns:
            torch.Tensor: The inferred state.
        """
        last_action_onehot = torch.nn.functional.one_hot(last_action, num_classes=5).float()
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        if last_action_onehot.dim() == 1:
            last_action_onehot = last_action_onehot.unsqueeze(0)

        x = torch.cat([observation, last_action_onehot], dim=-1)
        return self.network(x)

class CriticNet(nn.Module):
    """
    A neural network that estimates the Expected Free Energy (EFE) for each possible action,
    given an inferred state. The agent uses these EFE values to select its next action.
    """
    def __init__(self, input_size=2, hidden_size=64, output_size=5):
        """
        Initializes the critic network.
        Args:
            input_size (int): Size of the input (inferred state).
            hidden_size (int): Size of the hidden layers.
            output_size (int): Size of the output (EFE for each action).
        """
        super(CriticNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, state):
        """
        Performs a forward pass through the network.
        Args:
            state (torch.Tensor): The inferred state from the posterior network.
        Returns:
            torch.Tensor: The estimated EFE for each action.
        """
        return self.network(state)

# --- Active Inference Agent ---

class ActiveInferenceAgent:
    """
    The Active Inference agent. It uses a posterior network to infer its state and a critic
    network to evaluate actions based on Expected Free Energy. It learns by minimizing
    both VFE (via the posterior) and EFE prediction error (via the critic).
    """
    def __init__(self, state_size=2, action_size=5, lr=1e-4, gamma=0.99, zeta=1.0):
        """
        Initializes the agent.
        Args:
            state_size (int): The dimensionality of the state space.
            action_size (int): The number of possible actions.
            lr (float): The learning rate for the optimizers.
            gamma (float): The discount factor for future EFE.
            zeta (float): The precision parameter for action selection (inverse temperature).
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.zeta = zeta
        self.memory = deque(maxlen=10000)

        self.posterior_net = PosteriorNet(input_size=state_size + action_size, output_size=state_size)
        self.critic_net = CriticNet(input_size=state_size, output_size=action_size)
        self.target_critic_net = CriticNet(input_size=state_size, output_size=action_size)
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())
        self.target_critic_net.eval()

        self.posterior_optimizer = optim.Adam(self.posterior_net.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def infer_state(self, observation, last_action):
        """
        Infers the current hidden state given the latest observation and the previous action.
        Args:
            observation (np.ndarray): The noisy observation from the environment.
            last_action (int): The last action taken by the agent.
        Returns:
            np.ndarray: The agent's inferred belief about its current state.
        """
        obs_tensor = torch.FloatTensor(observation)
        action_tensor = torch.LongTensor([last_action])
        with torch.no_grad():
            inferred_state = self.posterior_net(obs_tensor, action_tensor)
        return inferred_state.squeeze(0).numpy()

    def choose_action(self, inferred_state):
        """
        Selects an action by sampling from a softmax distribution over the negative EFE values.
        Args:
            inferred_state (np.ndarray): The agent's inferred state.
        Returns:
            int: The selected action.
        """
        state_tensor = torch.FloatTensor(inferred_state).unsqueeze(0)
        with torch.no_grad():
            efe_values = self.critic_net(state_tensor)

        action_probs = torch.nn.functional.softmax(-self.zeta * efe_values, dim=-1)
        action = torch.multinomial(action_probs, 1).item()
        return action

    def update_target_critic(self):
        """Copies the weights from the main critic network to the target critic network."""
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())

    def remember(self, true_s_t, a_t_minus_1, a_t, true_s_t_plus_1, obs_t, obs_t_plus_1, goal):
        """
        Stores an experience transition in the replay memory.
        """
        self.memory.append((true_s_t, a_t_minus_1, a_t, true_s_t_plus_1, obs_t, obs_t_plus_1, goal))

    def replay(self, batch_size):
        """
        Trains the posterior and critic networks using a batch of experiences from memory.
        Args:
            batch_size (int): The number of experiences to sample from memory.
        Returns:
            tuple: A tuple containing the posterior loss and critic loss for the batch.
        """
        if len(self.memory) < batch_size:
            return None, None

        minibatch = random.sample(self.memory, batch_size)

        true_s_ts, a_t_minus_1s, a_ts, true_s_t_plus_1s, obs_ts, obs_t_plus_1s, goals = zip(*minibatch)

        true_s_t = torch.FloatTensor(np.array(true_s_ts))
        a_t_minus_1 = torch.LongTensor(np.array(a_t_minus_1s))
        obs_t = torch.FloatTensor(np.array(obs_ts))

        true_s_t_plus_1 = torch.FloatTensor(np.array(true_s_t_plus_1s))
        a_t = torch.LongTensor(np.array(a_ts))
        obs_t_plus_1 = torch.FloatTensor(np.array(obs_t_plus_1s))
        goal_tensor = torch.FloatTensor(np.array(goals))

        # --- Update Posterior Net (Minimize VFE Complexity) ---
        inferred_s_t = self.posterior_net(obs_t, a_t_minus_1)
        posterior_loss = self.loss_fn(inferred_s_t, true_s_t)

        self.posterior_optimizer.zero_grad()
        posterior_loss.backward()
        self.posterior_optimizer.step()

        # --- Update Critic Net (Minimize EFE Prediction Error) ---
        with torch.no_grad():
            inferred_s_t_plus_1 = self.posterior_net(obs_t_plus_1, a_t)

            epistemic_gain = -0.5 * torch.sum((inferred_s_t_plus_1 - true_s_t_plus_1)**2, dim=1)
            pragmatic_gain = -0.5 * torch.sum((obs_t_plus_1 - goal_tensor)**2, dim=1)

            one_step_efe = epistemic_gain + pragmatic_gain

            next_efe_values = self.target_critic_net(inferred_s_t_plus_1)
            next_efe, _ = torch.max(next_efe_values, dim=1)

            target_efe = one_step_efe + self.gamma * next_efe

        predicted_efe_values = self.critic_net(inferred_s_t.detach())
        predicted_efe = predicted_efe_values.gather(1, a_t.unsqueeze(1)).squeeze()

        critic_loss = self.loss_fn(predicted_efe, target_efe)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return posterior_loss.item(), critic_loss.item()

# --- Visualization ---

def plot_trajectory(trajectory, grid_size, goal_location):
    """
    Plots the agent's trajectory from the final episode.
    Args:
        trajectory (list): A list of the agent's coordinates during the episode.
        grid_size (int): The size of the grid.
        goal_location (np.ndarray): The coordinates of the goal.
    """
    path = np.array(trajectory)
    plt.figure(figsize=(8, 8))
    plt.plot(path[:, 1], path[:, 0], marker='o', linestyle='-', label='Agent Path')
    plt.plot(path[0, 1], path[0, 0], 'go', markersize=15, label='Start')
    plt.plot(path[-1, 1], path[-1, 0], 'ro', markersize=15, label='End')
    plt.plot(goal_location[1], goal_location[0], 'bx', markersize=20, markeredgewidth=3, label='Goal')

    plt.grid(True)
    plt.xticks(np.arange(-0.5, grid_size, 1), labels=np.arange(0, grid_size+1))
    plt.yticks(np.arange(-0.5, grid_size, 1), labels=np.arange(0, grid_size+1))
    plt.xlim([-0.5, grid_size - 0.5])
    plt.ylim([-0.5, grid_size - 0.5])
    plt.gca().invert_yaxis()
    plt.title('Agent Trajectory (Final Episode)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.savefig('trajectory.png')
    plt.close()
    print("Trajectory plot saved to trajectory.png")

def plot_losses(posterior_losses, critic_losses):
    """
    Plots the learning curves (losses) for the posterior and critic networks.
    Args:
        posterior_losses (list): A list of loss values for the posterior network.
        critic_losses (list): A list of loss values for the critic network.
    """
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(posterior_losses)
    plt.title('Posterior Network Loss (VFE Complexity)')
    plt.xlabel('Training Step')
    plt.ylabel('MSE Loss')

    plt.subplot(1, 2, 2)
    plt.plot(critic_losses)
    plt.title('Critic Network Loss (EFE Error)')
    plt.xlabel('Training Step')
    plt.ylabel('MSE Loss')

    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.close()
    print("Learning curves plot saved to learning_curves.png")


if __name__ == '__main__':
    # --- Simulation Setup ---
    NUM_EPISODES = 500
    MAX_STEPS_PER_EPISODE = 100
    BATCH_SIZE = 64
    TARGET_UPDATE_FREQ = 10

    env = GridWorld()
    agent = ActiveInferenceAgent()

    all_trajectories = []
    posterior_losses = []
    critic_losses = []

    print("--- Starting Simulation ---")

    for episode in range(NUM_EPISODES):
        obs_t, true_s_t = env.reset()
        a_t_minus_1 = np.random.randint(env.action_space_size)

        trajectory = [true_s_t]

        for step in range(MAX_STEPS_PER_EPISODE):
            inferred_s_t = agent.infer_state(obs_t, a_t_minus_1)
            a_t = agent.choose_action(inferred_s_t)

            obs_t_plus_1, true_s_t_plus_1 = env.step(a_t)

            agent.remember(true_s_t, a_t_minus_1, a_t, true_s_t_plus_1, obs_t, obs_t_plus_1, env.goal_location)

            obs_t = obs_t_plus_1
            true_s_t = true_s_t_plus_1
            a_t_minus_1 = a_t

            trajectory.append(true_s_t)

            p_loss, c_loss = agent.replay(BATCH_SIZE)
            if p_loss and c_loss:
                posterior_losses.append(p_loss)
                critic_losses.append(c_loss)

        all_trajectories.append(trajectory)

        if (episode + 1) % TARGET_UPDATE_FREQ == 0:
            agent.update_target_critic()
            print(f"Episode {episode + 1}/{NUM_EPISODES} - Target network updated.")

    print("\n--- Simulation Finished ---\n")

    # --- Visualization ---
    plot_trajectory(all_trajectories[-1], env.size, env.goal_location)
    plot_losses(posterior_losses, critic_losses)
