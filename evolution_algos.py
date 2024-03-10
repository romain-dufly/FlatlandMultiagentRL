import numpy as np
import math
from observation_utils import normalize_observation


def cem_uncorrelated(objective_function,
                     mean_array,
                     var_array,
                     max_iterations=500,
                     sample_size=50,
                     elite_frac=0.2,
                     print_every=10,
                     success_score=float("inf"),
                     num_evals_for_stop=None,
                     hist_dict=None):
    """Cross-entropy method.

    Params
    ======
        objective_function (function): the function to maximize
        mean_array (array of floats): the initial proposal distribution (mean vector)
        var_array (array of floats): the initial proposal distribution (variance vector)
        max_iterations (int): number of training iterations
        sample_size (int): size of population at each iteration
        elite_frac (float): rate of top performers to use in update with elite_frac ∈ ]0;1]
        print_every (int): how often to print average score
        hist_dict (dict): logs
    """
    assert 0. < elite_frac <= 1.

    n_elite = math.ceil(sample_size * elite_frac)
    var_array = np.diag(var_array)
    
    for it in range(max_iterations):

        print("iteration : ", it)

        samples = np.random.multivariate_normal(mean = mean_array, cov = var_array, size = sample_size)
        eval_samples = []
        for sample in samples:
            eval_samples.append([sample,objective_function(sample)])
        eval_samples.sort(key = lambda sample : sample[1])
        elite = []
        elite_eval = []
        for i in range(n_elite):
            elite.append(eval_samples[i][0])
            elite_eval.append(eval_samples[i][1])
        average_reward = np.mean([sample[1] for sample in eval_samples])
        hist_dict[it] = [average_reward] + [mu for mu in mean_array] + [var for var in np.diag(var_array)]

        if it%print_every == 0:
            print("sorted samples : ", [sample[1] for sample in eval_samples])
            print(average_reward)
        if average_reward < success_score:
           print("Success! Algorithm converged in ", it, " iterations")
           break
        mean_array = np.mean(elite, axis=0)
        var_array = np.cov(elite, rowvar=False)

    return mean_array



def saes_1_1(objective_function,
             x_array,
             sigma_array,
             max_iterations=500,
             tau=None,
             print_every=10,
             success_score=float("inf"),
             num_evals_for_stop=None,
             hist_dict=None):
    if tau is None:
        tau = 0
    for it in range(max_iterations):
        f_x = objective_function(x_array)
        if it%print_every == 0:
            print(f_x)
        if f_x < success_score:
            break

        sigma_array_prim = np.copy(sigma_array) * np.exp(tau*np.random.normal())
        x_array_prim = np.copy(x_array) + np.copy(sigma_array_prim) * np.random.normal()
        f_x_prim = objective_function(x_array_prim)
        if f_x_prim <= f_x:
            x_array = np.copy(x_array_prim)
            sigma_array  = np.copy(sigma_array_prim)
        hist_dict[it] = [f_x] + [mu for mu in x_array] + [var for var in sigma_array]



    return x_array


class ObjectiveFunction:

    def __init__(self, env, policy, observation_tree_depth, observation_radius, num_episodes=1, max_time_steps=float('inf'), minimization_solver=True):
        self.ndim = policy.num_params  # Number of dimensions of the parameter (weights) space
        self.env = env
        self.policy = policy
        self.num_episodes = num_episodes
        self.max_time_steps = max_time_steps
        self.minimization_solver = minimization_solver
        self.observation_tree_depth = observation_tree_depth
        self.observation_radius = observation_radius

        self.num_evals = 0


    def eval(self, policy_params, num_episodes=None, max_time_steps=None, render=False):
        """Evaluate a policy"""

        #print("Evaluation function called")

        self.num_evals += 1

        if num_episodes is None:
            num_episodes = self.num_episodes

        if max_time_steps is None:
            max_time_steps = self.max_time_steps

        average_total_rewards = 0

        for i_episode in range(num_episodes):

            total_rewards = 0.
            state, info = self.env.reset()

            for t in range(max_time_steps):
                if render:
                    self.env.render_wrapper.render()
                
                agent_obs = [None] * self.env.get_num_agents()
                for agent in self.env.get_agent_handles():
                    if state[agent]:
                        agent_obs[agent] = normalize_observation(state[agent], tree_depth=self.observation_tree_depth, observation_radius=self.observation_radius)
                actions = {}
                for agent in self.env.get_agent_handles():
                    actions[agent] = self.policy.act(agent_obs[agent], policy_params)
                
                state, all_rewards, done, info = self.env.step(actions)
                #print("step : ", t)
                #print("rewards : ", all_rewards)
                #print("actions : ", actions)
                total_rewards += sum(all_rewards.values())

                if done['__all__']:
                    #print("All agents reached their targets!")
                    #print(all_rewards)
                    break

            average_total_rewards += float(total_rewards) / num_episodes
            #print("Test Episode {0}: Total Reward = {1}".format(i_episode, total_rewards))

            if render:
                print("Test Episode {0}: Total Reward = {1}".format(i_episode, total_rewards))

        if self.minimization_solver:
            average_total_rewards *= -1.

        return average_total_rewards   # Optimizers do minimization by default...


    def __call__(self, policy_params, num_episodes=None, max_time_steps=None, render=False):
        return self.eval(policy_params, num_episodes, max_time_steps, render)