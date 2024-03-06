from observation_utils import normalize_observation
from flatland.envs.step_utils.states import TrainState

import numpy as np


def test_policy(env, policy, n_eval_episodes, obs_params, seeds=None):

    # Check len(seeds) == n_eval_episodes
    if seeds is not None:
        assert len(seeds) == n_eval_episodes
    else :
        seeds = list(range(n_eval_episodes))

    tree_depth = obs_params['observation_tree_depth']
    observation_radius = obs_params['observation_radius']

    scores = []
    completions = []
    nb_steps = []

    for seed in seeds:

        obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True, random_seed=seed)
        
        max_steps = env._max_episode_steps
        action_dict = dict()
        agent_obs = [None] * env.get_num_agents()    

        score = 0.0

        final_step = 0

        for step in range(max_steps):
            for agent in env.get_agent_handles():
                if obs[agent]:
                    agent_obs[agent] = normalize_observation(obs[agent], tree_depth=tree_depth, observation_radius=observation_radius)

                action = 0
                if info['action_required'][agent]:
                    action = policy.act(agent_obs[agent], eps=0.0)
                action_dict.update({agent: action})

            obs, all_rewards, done, info = env.step(action_dict)

            for agent in env.get_agent_handles():
                score += all_rewards[agent]

            final_step = step

            if done['__all__']:
                break

        normalized_score = score / (max_steps * env.get_num_agents())
        scores.append(normalized_score)

        tasks_finished = sum([agent.state == TrainState.DONE for agent in env.agents])
        completion = tasks_finished / max(1, env.get_num_agents())
        completions.append(completion)

        nb_steps.append(final_step)

    print("\t✅ Eval: score {:.3f} done {:.1f}%".format(np.mean(scores), np.mean(completions) * 100.0))

    return scores, completions, nb_steps, env.seed_history

def render_one_test(env, policy, obs_params, seed=0):
    obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True, random_seed=seed)