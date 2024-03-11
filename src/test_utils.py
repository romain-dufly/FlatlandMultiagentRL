
from flatland.utils.rendertools import RenderTool
from flatland.envs.step_utils.states import TrainState

import numpy as np
import PIL
import imageio
from IPython.display import Image, clear_output

from src.observation_utils import *

# Adapted for the labs code and code from flatland documentation
class RenderWrapper:
    '''Wrapper for the flatland environment to render it in a Jupyter Notebook
    Args:
        env (flatland.envs.rail_env.RailEnv): Environment to render
        real_time_render (bool): If True, the environment will be rendered in real time
        force_gif (bool): If True, calling make_gif will generate a gif from the rendered frames'''
    def __init__(self, env, real_time_render=False, force_gif=False):
        self.env_renderer = RenderTool(env, gl="PILSVG")
        self.real_time_render = real_time_render
        self.force_gif = force_gif
        self.reset()

    def reset(self):
        self.images = []

    def render(self):

        self.env_renderer.render_env(show_observations=False)

        image = self.env_renderer.get_image()
        pil_image = PIL.Image.fromarray(image)

        if self.real_time_render :
            clear_output(wait=True)
            display(pil_image)

        if self.force_gif:
            self.images.append(pil_image)

    def make_gif(self, filename="render"):
        if self.force_gif:
            imageio.mimsave(filename + '.gif', [np.array(img) for i, img in enumerate(self.images) if i%1 == 0], duration=100, loop=0)
            return Image(open(filename + '.gif','rb').read())

def run_one_test(env, policy, seed, obs_params, env_renderer=None, LSTM=False) :

    tree_depth = obs_params['observation_tree_depth']
    observation_radius = obs_params['observation_radius']

    obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True, random_seed=seed)
    if env_renderer is not None:
        env_renderer.render()
    
    if LSTM:
        obs_list = normalize_cutils(obs, env)
        
    max_steps = env._max_episode_steps
    action_dict = dict()
    agent_obs = [None] * env.get_num_agents()    

    score = 0.0

    final_step = 0

    for step in range(max_steps):
        for agent in env.get_agent_handles():
            if obs[agent]:
                if LSTM:
                    agent_obs[agent] = get_features([individual_from_obs_list(obs_list[0], agent)])
                else:
                    agent_obs[agent] = normalize_observation(obs[agent], tree_depth=tree_depth, observation_radius=observation_radius)
            
            action = 0
            if info['action_required'][agent]:
                action = policy.act(agent_obs[agent], eps=0.0)
            action_dict.update({agent: action})

        obs, all_rewards, done, info = env.step(action_dict)
        
        if env_renderer is not None:
            env_renderer.render()

        for agent in env.get_agent_handles():
            score += all_rewards[agent]

        final_step = step

        if done['__all__']:
            break

    normalized_score = score / (max_steps * env.get_num_agents())

    tasks_finished = sum([agent.state == TrainState.DONE for agent in env.agents])
    completion = tasks_finished / max(1, env.get_num_agents())

    return normalized_score, completion, final_step

def test_policy(env, policy, n_eval_episodes, obs_params, seeds=None, LSTM=False):

    # Check len(seeds) == n_eval_episodes
    if seeds is not None:
        assert len(seeds) == n_eval_episodes
    else :
        seeds = list(range(1, n_eval_episodes+1))

    scores = []
    completions = []
    nb_steps = []

    for seed in seeds:

        normalized_score, completion, final_step = run_one_test(env, policy, seed, obs_params, env_renderer=None, LSTM=LSTM)

        scores.append(normalized_score)
        completions.append(completion)
        nb_steps.append(final_step)

    print("\t✅ Eval: score {:.3f} done {:.1f}%".format(np.mean(scores), np.mean(completions) * 100.0))

    return scores, completions, nb_steps, env.seed_history[-n_eval_episodes:]

def render_one_test(env, policy, obs_params, real_time_render = True, force_gif=False, seed=1, LSTM=False):
    obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True, random_seed=seed)
    env_renderer = RenderWrapper(env, real_time_render, force_gif)
    run_one_test(env, policy, seed, obs_params, env_renderer, LSTM)
    return env_renderer