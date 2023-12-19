import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import warnings
from typing import Union
from utils import ReplayBuffer, get_env, run_episode
import random

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


torch.manual_seed(0)        #can tweak that

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
        '''act_fun = {
            "ReLU": torch.nn.ReLU,
            "GELU": torch.nn.GELU,
            "": torch.nn.SELU,
        }[activation]'''
        self.forward_pass = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_size),
            nn.BatchNorm1d(hidden_size),                                   #to avoid vanishing and exploiding gradient problem 
            nn.GELU(),
            # Make as many Linear + ReLU hidden layers as given as an argument
            *[
                nn.Sequential(
                    nn.Linear(in_features=hidden_size, out_features=hidden_size),
                    nn.BatchNorm1d(hidden_size),                           # same reason as above, currently does not work due to batch size 1, ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 256])
                    nn.GELU())
                for _ in range(hidden_layers)
            ],
            nn.Linear(in_features=hidden_size, out_features=output_dim),
        )
        #----------------------------------------------------------------------




    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass for the neural network you have defined.
        #----------------------------------------------------------------------
        return self.forward_pass(s)
        #----------------------------------------------------------------------

    
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
        #----------------------------------------------------------------------
        self.network = NeuralNetwork(
            input_dim=self.state_dim,
            output_dim=self.action_dim*2, # S: outpus mean and std for each prediction
            hidden_size=self.hidden_size,
            hidden_layers=self.hidden_layers,
            activation="").to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr = self.actor_lr)
        #print(f"actor act dim: {self.action_dim}")
        #----------------------------------------------------------------------


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
        #action , log_prob = torch.zeros(state.shape[0]), torch.ones(state.shape[0])
        # TODO: Implement this function which returns an action and its log probability.
        # If working with stochastic policies, make sure that its log_std are clamped 
        # using the clamp_log_std function.
        #----------------------------------------------------------------------
        # S: tensor shape, cuda activation
        state = state.to(self.device)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, dim=0)
        
        network_output = self.network(state)
        if len(network_output.shape) == 1:
            network_output = torch.unsqueeze(network_output, dim=0)


        action_mean = network_output[:, 0] #torch.clamp(network_output[:, :self.action_dim], -1, 1)
        action_log_std = network_output[:, 1]
        
        action_std = torch.exp(self.clamp_log_std(action_log_std))
        #debugging:
        if torch.any(torch.isnan(action_mean)) or torch.any(torch.isnan(action_std)):
            print(f"network input {state}")
            print(f"network output {network_output}")
            print(f"action mean and clamped std: {action_mean} {action_std}")
        print(f"action mean and std: {action_mean} {action_std}")
        
        #print("state shape", state.shape)
        action_dist = Normal(action_mean, action_std)
        if deterministic:
            action_mean_tanh = torch.nn.functional.tanh(action_mean)
            action = action_mean_tanh
            #action picked deterministic => prob 1, log_prob 0
            log_prob = torch.ones((state.shape[0],1))
            
        else:
            action_raw = action_dist.rsample()
            #tweak: clamp or tanh
            action = torch.tanh(action_raw) #torch.clamp(action, -1,1)]
            
            #subtract random jitter from log_prob
            jitter = torch.log(1 - pow(action,2) + 1e-6)
            log_prob = action_dist.log_prob(action_raw) - jitter
                        
            #print(action.shape, "action shape stochastic")

        action = torch.reshape(action, (state.shape[0],1))
        log_prob = torch.reshape(log_prob, (state.shape[0],1))
        #print(state.shape[0])
        #print(action.shape, log_prob.shape)
        #----------------------------------------------------------------------
        assert (action.shape  == torch.Size((self.action_dim,)) and \
                log_prob.shape  == torch.Size((self.action_dim,))) or (action.shape  == torch.Size((state.shape[0], 1)) and \
                log_prob.shape == torch.Size((state.shape[0], 1))), 'Incorrect shape for action and logprob'
        return action, log_prob


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
        self.setup_critic()

    def setup_critic(self):
        # TODO: Implement this function which sets up the critic(s). Take a look at the NeuralNetwork 
        # class in utils.py. Note that you can have MULTIPLE critic networks in this class.
        #----------------------------------------------------------------------
        #L: todo later: several critics
        # Create self.networks many networks.
        # The range(5) is completely arbitrary. Changing it should change 
        #  predictive performace and computational load.
        self.network = NeuralNetwork(
                input_dim=self.action_dim+self.state_dim,       #C: Q(x,a)
                output_dim=1, #S: Reward is probably 1-d
                hidden_layers=self.hidden_layers,
                hidden_size=self.hidden_size,
                activation="").to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr = self.critic_lr)
        #----------------------------------------------------------------------

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

        #L: ours
        #----------------------------------------------------------------------
        self.alpha = TrainableParameter(1e1, 0.05, train_param=True, device=self.device)
        #tweak: 
        #self.gamma= TrainableParameter(0.9, 3e-8, train_param=True, device=self.device)   #no loss which shows improvement 
        self.iter=1
        #----------------------------------------------------------------------
        self.setup_agent()

    def setup_agent(self):
        # TODO: Setup off-policy agent with policy and critic classes. 
        # Feel free to instantiate any other parameters you feel you might need.
        #----------------------------------------------------------------------
        
        actor_init = (256, 8, 1e-9, self.state_dim, self.action_dim, self.device)   #learning rates
        critic_init = (256, 4, 1e-9,self.state_dim, self.action_dim, self.device)
        self.actor = Actor(*actor_init)
        self.critic = Critic(*critic_init)
        
        #tweak: use of extra Critics
        self.critic2 = Critic(*critic_init) #not used yet
        self.critic_target = Critic(*critic_init)
        #tweak: use extra critics
        self.critic_target2 = Critic(*critic_init) #not used yet

        #----------------------------------------------------------------------

    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode. 
                    You can find it useful if you want to sample from deterministic policy.
        :return: np.ndarray,, action to apply on the environment, shape (1,)
        """
        # TODO: Implement a function that returns an action from the policy for the state s.
        #action = np.random.uniform(-1, 1, (1,)) #L: assumed placeholder
        #----------------------------------------------------------------------
        deterministic = not train #potential tweak: for now this, might change, good score version has it like this
        #print(f"state shape {s.shape}")
        with torch.inference_mode():
            self.actor.network.eval()
            action, _ = self.actor.get_action_and_log_prob(torch.tensor(s), deterministic)
        action = action.cpu().numpy()[0]
        #Tweak: changed it to this
        self.actor.network.train()
        #print(f"get_action: current state: {s[0]}; our action: {action}")
        #----------------------------------------------------------------------
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
        loss.mean().backward() #L: gradient
        object.optimizer.step() #L: addition

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
        #----------------------------------------------------------------------
        #not sure whether anything is needed here
        print("grad policy net start train_agent", self.actor.network.forward_pass[0].weight.grad)
        #----------------------------------------------------------------------

        # Batch sampling
        batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch = batch
        # TODO: Implement Critic(s) update here.
        #----------------------------------------------------------------------
        # S: sample next action
        GAMMA = 0.5 #tweak: discount factor -> C: trainable parameter as well, maybe optimize as well
        #S: define critic loss
        with torch.no_grad():
            state_action = torch.hstack((a_batch, s_batch))
            # C: tweak maybe implement epsilon for exploitation exploration in deterministic
            # deterministic = True
            
            # constant= 100                  #equals to 10 reward prints, self.iter==200 -> 1 reward epoch            
            # epsilon= min(1,constant/self.iter)
            # self.iter= self.iter+1
            # deterministic=random.choices([False, True], weights=[epsilon, 1-epsilon], k=1)[0]

            deterministic= False

            s_prime_action, s_prime_entropies = self.actor.get_action_and_log_prob(s_prime_batch, deterministic)
            state_action_prime = torch.hstack((s_prime_action, s_prime_batch))
            #prime_action, prime_entropy = self.actor.get_action_and_log_prob(s_prime_batch)
            # prime_reward = self.critic_target(torch.hstack(prime_action, s_prime_batch))
            s_prime_reward_critic = self.critic_target.network(state_action_prime)
            #tweak: self.gamma.get_param().detach()
            y = r_batch +  GAMMA* s_prime_reward_critic - self.alpha.get_param().detach() * s_prime_entropies
            s_reward_critic_new = self.critic.network(state_action)
        
        #L: debug
        do_a_pred_comp = False
        if do_a_pred_comp:
            next_a_pred_real = list(zip(s_prime_action[:-1], a_batch[1:]))
            next_a_pred_real = [(np.round(a.item(),2), np.round(b.item(),2)) for a,b in next_a_pred_real] 
            print(f"pred next, real actions: {next_a_pred_real}")
            
        
        # Implement Critc Update
        critic_loss = torch.nn.functional.mse_loss(y, self.critic.network(state_action))
        #print(f"critic loss: {critic_loss}")
        
        self.run_gradient_update_step(self.critic, critic_loss)
        #----------------------------------------------------------------------
        # TODO: Implement Policy update here
        #----------------------------------------------------------------------
        
        #tweak: added this, may be neutral change
        #self.actor.optimizer.zero_grad()
        #self.critic.optimizer.zero_grad()
        
        action_new, entropies = self.actor.get_action_and_log_prob(s_batch, deterministic)
        s_reward_critic_new = self.critic.network(torch.hstack((action_new, s_batch)))

        #policy gradient
        actor_loss = torch.mean(self.alpha.get_param().detach() * entropies - s_reward_critic_new)
        
        self.run_gradient_update_step(self.actor, actor_loss)
        #self.critic.optimizer.zero_grad()

        #----------------------------------------------------------------------

        #L: update alpha:
        #----------------------------------------------------------------------
        #tweak: potential hyperparameter test it!
        target_entropy = 1
        alpha_loss = -self.alpha.get_param() * torch.mean(entropies.detach() + target_entropy)
        
        self.run_gradient_update_step(self.alpha, alpha_loss)
        # S: target critic update
        exp_learning_factor = 0.5
        self.critic_target_update(self.critic.network, self.critic_target.network, exp_learning_factor, True)
        #----------------------------------------------------------------------
        #print(f"critic: {np.round(critic_loss.item(),4)}, actor: {np.round(actor_loss.item(),4)}, alpha: {np.round(alpha_loss.item(),4)}")
        print("grad policy net end  train_agent", self.actor.network.forward_pass[0].weight.grad)

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
