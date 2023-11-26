"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
# import additional ...
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, DotProduct, ConstantKernel

np.random.seed(1)

# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA


# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        #pass
        #4 arrays of same size storing xs, fs, vs and acquisition function values af
        #print("new class")
        self.xs = np.empty((1, 1))
        self.fs = np.empty((1, 1))
        self.vs = np.empty((1, 1))
        self.afs = np.array([])
        self.domain = np.atleast_2d(np.linspace(DOMAIN[0, 0], DOMAIN[0, 1], num=10000)).T
        self.lb = 1e5 #lambda penalty scaler for unsafe evaluations
        
        # We set it to None so that the code breaks if we actually try to index
        # an array with the "unset" value (None) of safe_point_idx, i.e. if we  
        # did not set it to something before usage.
        self.safe_point_idx = None
        
        #counters
        self.activ = 0
        self.add_dp = 0
        self.opt_activ = 0
        self.next_recomm = 0


        self.unsafe_tries = 0 #counter for unsafe tries where v > SAFETY_THRESHOLD
        f_kernel = Matern(nu=2.5, length_scale=.5)
        self.f_gpr = GaussianProcessRegressor(f_kernel, n_restarts_optimizer=10, alpha=0.15, normalize_y=True)
        v_kernel = ConstantKernel(4) + DotProduct(sigma_0=0) + Matern(nu=2.5, length_scale=.5)
        self.v_gpr = GaussianProcessRegressor(v_kernel, n_restarts_optimizer=10, alpha=0.0001, normalize_y=True)

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.

        
        #raise NotImplementedError
        #approach of randomly going left and right of x_init
        DO_TEST = False
        if DO_TEST:
            STEP = 0.05
            OFFSET = self.add_dp*STEP
            if self.xs[0] - OFFSET > 0 and self.add_dp % 2 == 1:
                OFFSET *= -1
                
            x_opt = self.xs[0] + OFFSET
            v_mean, v_std = self.v_gpr.predict([x_opt], return_std=True)
            v_pred = v_mean + v_std
            #print(f"v for offset {offset}: {v_pred}")
            return x_opt

        OFFSET = 0.06
        if self.add_dp == 1:
            return self.xs[0]+OFFSET
        if self.add_dp == 2:
            if self.xs[0] - OFFSET < 0.:
                return self.xs[0]+OFFSET*1.5
            else:
                return self.xs[0]-OFFSET

        x_opt = self.optimize_acquisition_function() #get the optimal next x
        v_mean, v_std = self.v_gpr.predict([[x_opt]], return_std=True)
        

        if self.add_dp == 3:
            v_pred = v_mean + 10*v_std
        else:
            v_pred = v_mean + 1.96*v_std
        print(f"v {v_pred} predicted, mean {v_mean}, std {v_std}")
        self.next_recomm += 1

        recommendation = x_opt
        # Make a random guess if we propose smth. that we already had.
        count = 0
        while recommendation in self.xs[:, 0] and count < 100 or (v_pred > 4 and count < 1) or (v_std > 0.15 and count<1):
            safe_point = self.xs[0, 0]
            interval = 0.75 * OFFSET if count < 10 else 0.3
            lower_bound = np.maximum(safe_point - interval, 0)
            upper_bound = np.minimum(safe_point + interval, 10)
            safer_domain = np.linspace(lower_bound, upper_bound, num=100)
            recommendation = np.random.choice(safer_domain)
            print(f"random sample: {recommendation}, count: {count}")
            count += 1

        #print(f"x recommended: {recommendation}")
        return recommendation

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """
        self.opt_activ += 1
        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        self.activ += 1
        x = np.atleast_2d(x)
        # TODO: Implement the acquisition function you want to optimize.
        #raise NotImplementedError
        
        
        v_mean, v_std = self.v_gpr.predict(x, return_std=True)
        if self.add_dp == 2:
            return -(v_mean + 3*v_std)
        
        f_mean, f_std = self.f_gpr.predict(x, return_std=True)

        v_pred = v_mean + 1.96*v_std
        weighted_penalty =  (self.lb * v_pred) #if v_pred > SAFETY_THRESHOLD else 0
        #print(f"weighted penalty: {weighted_penalty}") if weighted_penalty < 0 else None
        return (f_mean+f_std) - weighted_penalty


    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # TODO: Add the observed data {x, f, v} to your model.
        #raise NotImplementedError
        #filter incoming values for safe one
        
        
        
        if self.safe_point_idx==None and v[0] < SAFETY_THRESHOLD:
            self.safe_point_idx = self.add_dp
        
        #if v <= SAFETY_THRESHOLD*3:
        #print(f"adding {x} to array {self.xs}")
        if self.add_dp == 0:
            print(f"first v added: {v} with x {x}") #we know this is always safe
            self.xs = np.array([x])
            self.fs = np.array([f])
            self.vs = np.array([v])
        else:
            print(f"v {v} with x {x}")
            self.xs = np.vstack((self.xs, (x,)))
            #print(f"results in {self.xs}")
            #print(f"adding f {f} to array {self.fs}")
            self.fs = np.concatenate([self.fs, (f,)])
            #print(f"results in {self.fs}")
            #print(f"adding v {v} to array {self.vs}")
            self.vs = np.concatenate([self.vs, (v,)])
            #print(f"results in {self.vs}")

        #retrain
        self.f_gpr = self.f_gpr.fit(self.xs, self.fs)
        self.v_gpr = self.v_gpr.fit(self.xs, self.vs)

        self.add_dp += 1
        

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        #raise NotImplementedError
        #print("run get_solution")
        #print(f"activation function runs: {self.activ}, optimized_activation_functions {self.opt_activ}")
        #print(f"next_recommendations {self.next_recomm}, datapoints put in: {self.add_dp}")
        # Compute the predicitions
        f_predictions = self.f_gpr.predict(self.domain)
        v_predictions = self.v_gpr.predict(self.domain)
        
        
        # Select which preditions are (probably) within our safety threshold
        invalid_preds = v_predictions > SAFETY_THRESHOLD

        # Set all invalid predicitons to the worst value so we can take the 
        # maximum index later. This simplifies the handling.
        f_predictions[invalid_preds] = f_predictions.min() # Chris: easy baseline push/ prev best: self.lb= 1e-1 and using np.min(f_predictions)

        # Get the index of the maximum, i.e. the index of the optimal x in our
        # domain. The invalid predictions will always perform worse than the
        # valid ones due to our check above.
        index_opt = np.argmax(f_predictions)

        x_opt = self.domain[index_opt]
        
        return x_opt

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        pass


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.randn()
        cost_val = v(x) + np.randn()
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    #main()
    vs = [1,1,1,5,1,6]
    fs = [6,5,2,7,10,0]
    xs = [1,2,3,4,5,6]
    agent = BO_algo()
    agent.add_data_point(xs,fs,vs)
    print(agent.acquisition_function(xs))

