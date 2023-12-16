import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import warnings
from typing import Union
from utils import ReplayBuffer, get_env, run_episode

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class NeuralNetwork(nn.Module):
    '''
    This class implements a neural network with a variable number of hidden layers and hidden units.
    You may use this function to parametrize your policy and critic networks.
    '''
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, 
                                hidden_layers: int, activation: str):
        super(NeuralNetwork, self).__init__()

        # TODO: Implement this function which should define a neural network 
        # with a variable number of hidden layers and hidden units.
        # Here you should define layers which your network will use.
        act_fun = {
            "ReLU": torch.nn.ReLU,
            "GELU": torch.nn.GELU,
            "": torch.nn.SELU,
        }[activation]
        self.forward_pass = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_size),
            nn.ReLU(),
            # Make as many Linear + ReLU hidden layers as given as an argument
            *[
                nn.Sequential(
                    nn.Linear(in_features=hidden_size, out_features=hidden_size),
                    nn.ReLU())
                for _ in range(hidden_layers)
            ],
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=output_dim)
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass for the neural network you have defined.
        if not isinstance(s, torch.Tensor):
            s = torch.Tensor(s)
        return self.forward_pass(s)

# CURRENT GUESS: SHIT PPL SPEAK FOR POLICY
class Actor:
    def __init__(self,hidden_size: int, hidden_layers: int, actor_lr: float,
                state_dim: int = 3, action_dim: int = 1, device: torch.device = torch.device('cpu')):
        super(Actor, self).__init__()

        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.actor_lr = actor_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.setup_actor()

    def setup_actor(self):
        '''
        This function sets up the actor network in the Actor class.
        '''
        # TODO: Implement this function which sets up the actor network. 
        # Take a look at the NeuralNetwork class in utils.py. 
        self.network = torch.nn.Sequential(NeuralNetwork(
            input_dim=self.state_dim,
            output_dim=self.action_dim,
            hidden_size=self.hidden_size,
            hidden_layers=self.hidden_layers,
            activation=""),                      #currently just ReLu need to specify this later
            torch.nn.Tanh(), # The Tanh is to make the network only output in -1;1
        )
        self.optimizer = optim.AdamW(self.network.parameters(), lr = self.actor_lr)
        #print(f"actor act dim: {self.action_dim}")

    def clamp_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        '''
        :param log_std: torch.Tensor, log_std of the policy.
        Returns:
        :param log_std: torch.Tensor, log_std of the policy clamped between LOG_STD_MIN and LOG_STD_MAX.
        '''
        return torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

    def get_action_and_log_prob(self, state: torch.Tensor, 
                                deterministic: bool) -> (torch.Tensor, torch.Tensor):
        '''
        :param state: torch.Tensor, state of the agent
        :param deterministic: boolean, if true return a deterministic action 
                                otherwise sample from the policy distribution.
        Returns:
        :param action: torch.Tensor, action the policy returns for the state.
        :param log_prob: log_probability of the the action.
        '''
        assert state.shape == (3,) or state.shape[1] == self.state_dim, 'State passed to this method has a wrong shape'
        # Simon: I think state.shap[0] might be the batch dim.
        action , log_prob = torch.zeros(state.shape[0]), torch.ones(state.shape[0])
        # TODO: Implement this function which returns an action and its log probability.
        # If working with stochastic policies, make sure that its log_std are clamped 
        # using the clamp_log_std function.
        actions=[]
        for i in range(5):
            action = self.network.forward(state)                                     #getting possible actions from out network 
            actions.append(action)

        if deterministic:
           log_probabilities= torch.zeros(state.shape[0])                       #torch.log(probabilities) old one
           #idx = torch.argmax(log_probabilities)                                #taking the max arg of our log probabilities
           #action, log_prob= actions[idx], log_probabilities[idx]               #returning our max log probabilities and best action
           print(actions)
           action= torch.mean(torch.tensor(actions))
           print(f"action shape {action.shape}")
        else:
            #quite unsure with the results here, do not know why we need stdv yet 
            #currently implemented code for discrete action space 
            #if we want continous action spaces may need to change network structure

           mean_action= torch.mean(actions)
           print(f"mean action shape {mean_action.shape}")
           print(f"std action shape {std_action.shape}")
           log_std_unclamped = torch.log(torch.std(actions))

           log_std_action = self.clamp_log_std(log_std=log_std_unclamped)
           std_action = torch.exp(log_std_action)           

           dist = torch.distributions.normal.Normal(mean_action, std_action)              
           action = dist.sample()                                                #sample an action
           log_prob = dist.log_prob(action)
           print(f"log prob from dist {log_prob.shape}")                                      #get log prob of that (sampled) action
           #action = torch.multinomial(probabilities, num_samples=1).item()      #sampled an action
           #how to get the probability of sampled action?           
        print(f"action shape: {action.shape}; log_prob shape: {log_prob.shape}")

        assert action.shape == (state.shape[0], self.action_dim) and \
            log_prob.shape == (state.shape[0], self.action_dim), 'Incorrect shape for action or log_prob.'
        return action, log_prob

# CURRENT GUESS: SHIT PPL SPEAK FOR Q / VALUE FUNCTION
class Critic:
    def __init__(self, hidden_size: int, 
                 hidden_layers: int, critic_lr: int, state_dim: int = 3, 
                    action_dim: int = 1,device: torch.device = torch.device('cpu')):
        super(Critic, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.critic_lr = critic_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        #print(f"Critic state_dim: {state_dim}; action_dim: {action_dim}")
        self.setup_critic()

    def setup_critic(self):
        # TODO: Implement this function which sets up the critic(s). Take a look at the NeuralNetwork 
        # class in utils.py. Note that you can have MULTIPLE critic networks in this class.
        # Create self.networks many networks.
        # The range(5) is completely arbitrary. Changing it should change 
        #  predictive performace and computational load.
        '''self.networks = [
            NeuralNetwork(
                input_dim=self.state_dim,
                output_dim=self.action_dim,
                hidden_layers=self.hidden_layers,
                hidden_size=self.hidden_size,
                activation="")
            for _ in range(5)
        ]'''
        self.network = NeuralNetwork(
                input_dim=self.state_dim+self.action_dim,
                output_dim=1, # Reward is probably 1-d
                hidden_layers=self.hidden_layers,
                hidden_size=self.hidden_size,
                activation="")
        self.optimizer = optim.AdamW(self.network.parameters(), lr = self.critic_lr)

class TrainableParameter:
    '''
    This class could be used to define a trainable parameter in your method. You could find it 
    useful if you try to implement the entropy temerature parameter for SAC algorithm.
    '''
    def __init__(self, init_param: float, lr_param: float, 
                 train_param: bool, device: torch.device = torch.device('cpu')):
        
        self.log_param = torch.tensor(np.log(init_param), requires_grad=train_param, device=device)
        self.optimizer = optim.Adam([self.log_param], lr=lr_param)

    def get_param(self) -> torch.Tensor:
        return torch.exp(self.log_param)

    def get_log_param(self) -> torch.Tensor:
        return self.log_param


class Agent:
    def __init__(self):
        # Environment variables. You don't need to change this.
        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # [torque] in[-1,1]
        self.batch_size = 200
        self.min_buffer_size = 1000
        self.max_buffer_size = 100000
        # If your PC possesses a GPU, you should be able to use it for training, 
        # as self.device should be 'cuda' in that case.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))
        self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)
        
        self.setup_agent()

    def setup_agent(self):
        # TODO: Setup off-policy agent with policy and critic classes. 
        # Feel free to instantiate any other parameters you feel you might need.

        # (hidden_size, hidden_layers, lr, state_dim, action_dim, device)
        init_tuple = (100, 10, 1, self.state_dim, self.action_dim, self.device)
        self.actor = Actor(*init_tuple)
        self.critic = Critic(*init_tuple)

    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode. 
                    You can find it useful if you want to sample from deterministic policy.
        :return: np.ndarray,, action to apply on the environment, shape (1,)
        """
        # TODO: Implement a function that returns an action from the policy for the state s.
        # action = np.random.uniform(-1, 1, (1,))
        #torque/action between [-1,1] bc specified above
        
        #Chris: need to specify an optimizer within critic_target_update_step
        
        if train: 
            with torch.inference_mode():
                action, _ = self.actor.get_action_and_log_prob(s, True)
        else: 
            #S: evaluation mode, just forward pass -> actor picks directly
            with torch.inference_mode():
                # ATTENTION: the following code is at this moment this NOT correct!
                action, log_prob = self.actor.get_action_and_log_prob(s, True)
                print(f"log prob from get action {log_prob.shape}")    
        action = action.numpy()
        assert action.shape == (1,), 'Incorrect action shape.'
        assert isinstance(action, np.ndarray ), 'Action dtype must be np.ndarray' 
        return action

    @staticmethod
    def run_gradient_update_step(object: Union[Actor, Critic], loss: torch.Tensor):
        '''
        This function takes in a object containing trainable parameters and an optimizer, 
        and using a given loss, runs one step of gradient update. If you set up trainable parameters 
        and optimizer inside the object, you could find this function useful while training.
        :param object: object containing trainable parameters and an optimizer
        '''
        object.optimizer.zero_grad()
        loss.mean().backward()
        object.optimizer.step()

    def critic_target_update(self, base_net: NeuralNetwork, target_net: NeuralNetwork, 
                             tau: float, soft_update: bool):
        '''
        This method updates the target network parameters using the source network parameters.
        If soft_update is True, then perform a soft update, otherwise a hard update (copy).
        :param base_net: source network
        :param target_net: target network
        :param tau: soft update parameter
        :param soft_update: boolean to indicate whether to perform a soft update or not
        '''
        for param_target, param in zip(target_net.parameters(), base_net.parameters()):
            if soft_update:
                param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
            else:
                param_target.data.copy_(param.data)

    def train_agent(self):
        '''
        This function represents one training iteration for the agent. It samples a batch 
        from the replay buffer,and then updates the policy and critic networks 
        using the sampled batch.
        '''
        # TODO: Implement one step of training for the agent.
        # Hint: You can use the run_gradient_update_step for each policy and critic.
        # Example: self.run_gradient_update_step(self.policy, policy_loss)

        # Batch sampling
        batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch = batch

        #
        action_prediction, log_prob = self.actor.get_action_and_log_prob(s_batch, True)
        print(f"log prob from actor network {log_prob.shape}")
        #replay buffer tuple: (s_lst, a_lst, r_lst, s_prime_lst)             #save our quadruples
        policy_gradient = -log_prob* r_batch

        self.run_gradient_update_step(self.actor, policy_gradient)


        # TODO: Implement Critic(s) update here.
        # TODO SIMON: There is a conceptual error here. I do not know exactly. 
        # what actor is supposed to predict. Either it is the next state from 
        # the current state + action or it is the next action from the current 
        # state. Idk what we need s_prime for.
        # action_prediction = self.actor.network(s_batch)
        # self.run_gradient_update_step(self.critic, loss(action_prediction, a_batch))   #maybe implement the policy gradient 
        # TODO: Implement Policy update here
        # Simon: I think this part is correct, or at least good enough.
        loss = torch.nn.MSELoss()
        #print(state_action.shape)
        state_action = torch.hstack((s_batch, a_batch))
        critical_prediction = self.critic.network(state_action)
        self.run_gradient_update_step(self.critic, loss(critical_prediction, r_batch))
        


# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    TRAIN_EPISODES = 50
    TEST_EPISODES = 300

    # You may set the save_video param to output the video of one of the evalution episodes, or 
    # you can disable console printing during training and testing by setting verbose to False.
    save_video = False
    verbose = True

    agent = Agent()
    env = get_env(g=10.0, train=True)

    for EP in range(TRAIN_EPISODES):
        run_episode(env, agent, None, verbose, train=True)

    if verbose:
        print('\n')

    test_returns = []
    env = get_env(g=10.0, train=False)

    if save_video:
        video_rec = VideoRecorder(env, "pendulum_episode.mp4")
    
    for EP in range(TEST_EPISODES):
        rec = video_rec if (save_video and EP == TEST_EPISODES - 1) else None
        with torch.no_grad():
            episode_return = run_episode(env, agent, rec, verbose, train=False)
        test_returns.append(episode_return)

    avg_test_return = np.mean(np.array(test_returns))

    print("\n AVG_TEST_RETURN:{:.1f} \n".format(avg_test_return))

    if save_video:
        video_rec.close()
