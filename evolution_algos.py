import numpy as np
import math
from observation_utils import normalize_observation

from test_utils import RenderWrapper


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
            print("iteration : ", it)
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
        tau = 1. / math.sqrt(x_array.size)

    if hist_dict is not None:
        hist_dict['score'] = []
        hist_dict['mu1'] = []
        hist_dict['mu2'] = []
        hist_dict['mu3'] = []
        hist_dict['mu4'] = []
        hist_dict['sigma1'] = []
        hist_dict['sigma2'] = []
        hist_dict['sigma3'] = []
        hist_dict['sigma4'] = []

    for iteration in range(max_iterations):
            
            # Sample parameter vectors
            population = np.random.multivariate_normal(x_array, np.diag(sigma_array), 1)

            # Update the variance and mean of the population
            sigma_array_bis = sigma_array * np.exp(tau * np.random.normal(size=sigma_array.shape[0]))
            x_array_bis = x_array + sigma_array_bis * np.random.normal(size=x_array.shape[0])

            # Sample parameter vectors
            population_bis = np.random.multivariate_normal(x_array_bis, np.diag(sigma_array_bis), 1)
    
            # Evaluate the mean of the population
            score = objective_function(population[0])
            score_bis = objective_function(population_bis[0])

            # Update the mean and variance
            if score_bis < score:
                x_array = x_array_bis
                sigma_array = sigma_array_bis
    
            if hist_dict is not None:
                hist_dict['score'].append(score)
                hist_dict['mu1'].append(x_array[0])
                hist_dict['mu2'].append(x_array[1])
                hist_dict['mu3'].append(x_array[2])
                hist_dict['mu4'].append(x_array[3])
                hist_dict['sigma1'].append(sigma_array[0])
                hist_dict['sigma2'].append(sigma_array[1])
                hist_dict['sigma3'].append(sigma_array[2])
                hist_dict['sigma4'].append(sigma_array[3])
    
            if iteration % print_every == 0:
                print("Iteration {0}/{1}: Score = {2}".format(iteration, max_iterations, score))
    
            if score <= success_score:
                print(f"Success after {iteration} iterations!")
                break
    
            if num_evals_for_stop is not None and objective_function.num_evals >= num_evals_for_stop:
                print(f"Stop after {objective_function.num_evals} evaluations")
                break

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
        if render:
            render_wrapper = RenderWrapper(self.env, real_time_render=True, force_gif=False)
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
                    render_wrapper.render()
                
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