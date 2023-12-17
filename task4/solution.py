import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import warnings
from typing import Union
from utils import ReplayBuffer, get_env, run_episode

#tweak
torch.manual_seed(420)

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
        #----------------------------------------------------------------------
        self.forward_pass = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_size),
            #nn.BatchNorm1d(hidden_size),                                   #to avoid vanishing and exploiding gradient problem 
            nn.GELU(),
            # Make as many Linear + ReLU hidden layers as given as an argument
            *[
                nn.Sequential(
                    nn.Linear(in_features=hidden_size, out_features=hidden_size),
                    #nn.BatchNorm1d(hidden_size),                           # same reason as above, currently does not work due to batch size 1, ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 256])
                    nn.GELU())
                for _ in range(hidden_layers)
            ],
            nn.Linear(in_features=hidden_size, out_features=output_dim),
        )
        #----------------------------------------------------------------------

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass for the neural network you have defined.
        #----------------------------------------------------------------------
        if not isinstance(s, torch.Tensor):
            s = torch.Tensor(s)
        return self.forward_pass(s)
        #----------------------------------------------------------------------

class Actor:
    def __init__(self,hidden_size: int, hidden_layers: int, actor_lr: float,
                state_dim: int = 3, action_dim: int = 1, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
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
        self.network = NeuralNetwork(input_dim=self.state_dim,
                                       output_dim=2*self.action_dim,
                                         hidden_size=self.hidden_size, 
                                         hidden_layers=self.hidden_layers,
                                         activation="")
        self.network.to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr = self.actor_lr)

    def clamp_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        '''
        :param log_std: torch.Tensor, log_std of the policy.
        Returns:
        :param log_std: torch.Tensor, log_std of the policy clamped between LOG_STD_MIN and LOG_STD_MAX.
        '''
        return torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

    def get_action_and_log_prob(self, state: torch.Tensor, 
                                deterministic: bool = False) -> (torch.Tensor, torch.Tensor):
        '''
        :param state: torch.Tensor, state of the agent
        :param deterministic: boolean, if true return a deterministic action 
                                otherwise sample from the policy distribution.
        Returns:
        :param action: torch.Tensor, action the policy returns for the state.
        :param log_prob: log_probability of the the action.
        '''
        assert state.shape == (3,) or state.shape[1] == self.state_dim, 'State passed to this method has a wrong shape'
        action , log_prob = torch.zeros(state.shape[0]), torch.ones(state.shape[0])
        # TODO: Implement this function which returns an action and its log probability.
        # If working with stochastic policies, make sure that its log_std are clamped 
        # using the clamp_log_std function.

        state = state.to(self.device)
        output = self.network(state)
        N = state.shape[0]

        if state.shape == (3,):
            mean, log_std = output[0].reshape((self.action_dim,)), output[1].reshape((self.action_dim,) )
        else:
            mean, log_std = output[:, 0].reshape((N, self.action_dim)), output[:, 1].reshape((N, self.action_dim))
        log_std = self.clamp_log_std(log_std)
        dist = Normal(mean,torch.exp(log_std))
        if deterministic:
            action = torch.tanh(mean)
        else:
            #clamp the log_standard_deviation
            
            action = dist.rsample()
            action = torch.tanh(action)
            
        num_stab = 1e-6
        log_prob = dist.log_prob(action) - torch.log(1 - action.pow(2) + num_stab)

        assert (action.shape == (self.action_dim, ) and \
            log_prob.shape == (self.action_dim, ), 'Incorrect shape for action or log_prob.' ) or \
                ( action.shape[1] == self.action_dim and log_prob.shape[1] == self.action_dim )
    
        return action, log_prob

class Critic:
    def __init__(self, hidden_size: int, 
                 hidden_layers: int, critic_lr: int, state_dim: int = 3, 
                    action_dim: int = 1,device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(Critic, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.critic_lr = critic_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.setup_critic()

    def setup_critic(self):
        # TODO: Implement this function which sets up the critic(s). Take a look at the NeuralNetwork 
        # class in utils.py. Note that you can have MULTIPLE critic networks in this class.
        self.network = NeuralNetwork(input_dim = self.state_dim, output_dim=1, hidden_size=self.hidden_size, hidden_layers=self.hidden_layers, activation="")
        self.network.to(self.device)
        #init optimizer with custom critic lr
        self.optimizer = optim.Adam(self.network.parameters(),lr = self.critic_lr)


class TrainableParameter:
    '''
    This class could be used to define a trainable parameter in your method. You could find it 
    useful if you try to implement the entropy temerature parameter for SAC algorithm.
    '''
    def __init__(self, init_param: float, lr_param: float, 
                 train_param: bool, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        
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
        #print("Using device: {}".format(self.device))
        self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)
        
        self.setup_agent()

    def setup_agent(self):
        # TODO: Setup off-policy agent with policy and critic classes. 
        # Feel free to instantiate any other parameters you feel you might need.   
        #tweak 
        torch.manual_seed(69)

        self.hidden_layers = 1
        self.hidden_size = 10
        LR = 3e-2
        # Actor Net
        self.act_net = Actor(hidden_size = self.hidden_size,hidden_layers = self.hidden_layers,actor_lr = LR)
        
        # both critic Nets
        self.critic_net1 = Critic(state_dim = self.state_dim+self.action_dim,hidden_size = self.hidden_size, hidden_layers = self.hidden_layers,critic_lr = LR)
        self.critic_net2 = Critic(state_dim=self.state_dim+self.action_dim,hidden_size=self.hidden_size,hidden_layers=self.hidden_layers,critic_lr=LR)
        
        # both target Nets
        self.targetc_net1 = Critic(state_dim=self.state_dim+self.action_dim,hidden_size=self.hidden_size,hidden_layers=self.hidden_layers,critic_lr=LR)
        self.targetc_net2 = Critic(state_dim=self.state_dim+self.action_dim,hidden_size=self.hidden_size,hidden_layers=self.hidden_layers,    critic_lr=LR)
        
        # updating our nets
        self.critic_target_update(self.targetc_net1.network, self.critic_net1.network, 0., False)
        self.critic_target_update(self.targetc_net2.network, self.critic_net2.network, 0., False)

        self.Tau = 0.005
        self.gamma = 0.99
        self.alpha = 0.05
        self.learn_param = TrainableParameter(init_param=0.05, lr_param=0.01, train_param=True)

    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode. 
                    You can find it useful if you want to sample from deterministic policy.
        :return: np.ndarray,, action to apply on the environment, shape (1,)
        """
        # TODO: Implement a function that returns an action from the policy for the state s.
        s = torch.tensor(s)
        action, _ = self.act_net.get_action_and_log_prob(s, not train)
        
        action = action.cpu().clone().detach().numpy()

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

        # TODO: Implement Critic(s) update here.
        with torch.no_grad():
            next_sampled_action, next_sampled_log_prob = self.act_net.get_action_and_log_prob(s_prime_batch, False)

            state_action = torch.cat((s_prime_batch, next_sampled_action), dim = 1).to(self.device)

            next_reward_1 = self.targetc_net1.network(state_action)   
            next_reward_2 = self.targetc_net2.network(state_action)

            min_qf_next = torch.min(next_reward_1,next_reward_2) - self.alpha * next_sampled_log_prob

            next_q_value = r_batch + self.gamma * min_qf_next

        input_reward = torch.cat((s_batch, a_batch), dim = 1).to(self.device)
    
        reward_pred_1 = self.critic_net1.network(input_reward)
        reward_pred_2 = self.critic_net2.network(input_reward)

        q1_loss = nn.functional.mse_loss(reward_pred_1, next_q_value)  
        q2_loss = nn.functional.mse_loss(reward_pred_2,next_q_value)

        self.run_gradient_update_step(self.critic_net1, q1_loss)
        self.run_gradient_update_step(self.critic_net2, q2_loss)

        sampled_action, sampled_log_prob = self.act_net.get_action_and_log_prob(s_batch, False)

        input_policy = torch.cat((s_batch, sampled_action), dim = 1).to(self.device)
        Q1_pi = self.critic_net1.network(input_policy)
        Q2_pi = self.critic_net2.network(input_policy)
        min_q_pi = torch.min(Q1_pi, Q2_pi)
        
        # TODO: Implement Policy update here
        policy_loss = (self.alpha * (sampled_log_prob) - min_q_pi)

        # Policy step
        self.run_gradient_update_step(self.act_net, policy_loss)

        target_entropy = -1.
        
        sampled_log_prob_detached = sampled_log_prob.detach().clone()
        alpha = self.learn_param.get_param()
        alpha_loss = - alpha * sampled_log_prob_detached - alpha * target_entropy

        #self.update_step(loss=alpha_loss)
        self.learn_param.optimizer.zero_grad()
        alpha_loss.mean().backward()
        self.learn_param.optimizer.step()

        self.alpha = self.learn_param.get_param()

        self.critic_target_update(self.critic_net1.network, self.targetc_net1.network,self.Tau,True)
        self.critic_target_update(self.critic_net2.network, self.targetc_net2.network, self.Tau,True)
    
    def update_step(self, loss: torch.Tensor):
        self.learn_param.optimizer.zero_grad()
        loss.mean().backward()
        self.learn_param.optimizer.step()

# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    TRAIN_EPISODES = 50
    TEST_EPISODES = 50

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
