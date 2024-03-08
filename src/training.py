import numpy as np
import copy

from flatland.utils.rendertools import RenderTool
from observation_utils import *
from flatland.envs.step_utils.states import TrainState

def train_agent(env, policy, train_params, obs_params):
    # Observation parameters
    observation_tree_depth = obs_params['observation_tree_depth']
    observation_radius = obs_params['observation_radius']

    # Training parameters
    eps_start = train_params['eps_start']
    eps_end = train_params['eps_end']
    eps_decay = train_params['eps_decay']
    n_episodes = train_params['n_episodes']
    checkpoint_interval = train_params['checkpoint_interval']
    n_eval_episodes = train_params['n_eval_episodes']
    restore_replay_buffer = train_params['restore_replay_buffer']
    save_replay_buffer = train_params['save_replay_buffer']

    if train_params.get('LSTM'): # Observation normalization choice
        LSTM = train_params['LSTM']
    else:
        LSTM = False
    if train_params.get('centralized'): # Are agents trained separately or together
        centralized = train_params['centralized']
    else:
        centralized = False

    # Setup the environments
    _,_ = env.reset()
    train_env = env
    eval_env = env

    # Setup renderer
    if train_params['render']:
        env_renderer = RenderTool(train_env)

    # The action space of flatland is 5 discrete actions
    action_size = 5

    # Smoothed values used as target for hyperparameter tuning
    smoothed_normalized_score = -1.0
    smoothed_completion = 0.0

    # Loads existing replay buffer
    if restore_replay_buffer:
        try:
            policy.load_replay_buffer(restore_replay_buffer)
        except RuntimeError as e:
            print("\n Could't load replay buffer.")
            print(e)
            exit(1)

    print("\n💾 Replay buffer status: {}/{} experiences".format(len(policy.memory.memory), train_params['buffer_size']))

    if save_replay_buffer:
        print("Will save replay buffer. Doing so will quickly consume a lot of disk space.")

    n_agents = train_env.get_num_agents()
    print("\n🚉 Training {} trains on {}x{} grid for {} episodes, evaluating on {} episodes every {} episodes.\n".format(
        n_agents,
        train_env.width,
        train_env.height,
        n_episodes,
        n_eval_episodes,
        checkpoint_interval,
    ))

    for episode_idx in range(n_episodes):
        # Reset environment
        obs, info = train_env.reset(regenerate_rail=True, regenerate_schedule=True)

        # Init these values after reset()
        max_steps = train_env._max_episode_steps
        action_count = [0] * action_size
        action_dict = dict()
        agent_obs = [None] * n_agents
        agent_prev_obs = [None] * n_agents
        agent_prev_action = [2] * n_agents
        update_values = [False] * n_agents

        if train_params['render']:
            env_renderer.set_new_rail()

        score = 0
        nb_steps = 0
        actions_taken = []

        ################################################
        # Build initial agent-specific observations
        if LSTM:
            obs_list = normalize_cutils(obs, train_env)
        for agent in train_env.get_agent_handles():
            if obs[agent]:
                if LSTM:
                    agent_obs[agent] = get_features([individual_from_obs_list(obs_list[0], agent)])
                else:
                    agent_obs[agent] = normalize_observation(obs[agent], observation_tree_depth, observation_radius=observation_radius)
                agent_prev_obs[agent] = copy.deepcopy(agent_obs[agent])

        # Run episode
        for _ in range(max_steps):
            # Get all actions
            if centralized:
                pass
            else:
                for agent in train_env.get_agent_handles():
                    if info['action_required'][agent]:
                        update_values[agent] = True
                        action = policy.act(agent_obs[agent], eps=eps_start)

                        action_count[action] += 1
                        actions_taken.append(action)
                    else:
                        update_values[agent] = False
                        action = 0
                    action_dict.update({agent: action})

            # Environment step
            next_obs, all_rewards, done, info = train_env.step(action_dict)
            if LSTM:
                obs_list = normalize_cutils(next_obs, train_env)

            # Render an episode at some interval
            if train_params['render'] and episode_idx % checkpoint_interval == 0:
                env_renderer.render_env(
                    show=True,
                    frames=False,
                    show_observations=False,
                    show_predictions=False
                )

            # Update replay buffer and train agent
            for agent in train_env.get_agent_handles():
                if update_values[agent] or done['__all__']:
                    # Only learn from timesteps where somethings happened
                    policy.step(agent_prev_obs[agent], agent_prev_action[agent], all_rewards[agent], agent_obs[agent], done[agent])
                    agent_prev_obs[agent] = copy.deepcopy(agent_obs[agent])
                    agent_prev_action[agent] = action_dict[agent]
                score += all_rewards[agent]

                # Preprocess the new observations
                if next_obs[agent]:
                    if LSTM:
                        agent_obs[agent] = get_features([individual_from_obs_list(obs_list[0], agent)])
                    else:
                        agent_obs[agent] = normalize_observation(next_obs[agent], observation_tree_depth, observation_radius=observation_radius)

            if done['__all__']:
                break
        
        ################################################

        # Epsilon decay
        eps_start = max(eps_end, eps_decay * eps_start)

        # Collect information about training
        tasks_finished = sum([agent.state == TrainState.DONE for agent in train_env.agents])
        completion = tasks_finished / max(1, train_env.get_num_agents())
        normalized_score = score / (max_steps * train_env.get_num_agents())

        # if no actions were ever taken possibly due to malfunction and so 
        # - `actions_taken` is empty [], 
        # - `np.sum(action_count)` is 0
        # Set action probs to count
        if (np.sum(action_count) > 0):
            action_probs = action_count / np.sum(action_count)
        else:
            action_probs = action_count
        action_count = [1] * action_size

        # Set actions_taken to a list with single item 0
        if not actions_taken:
            actions_taken = [0]

        smoothing = 0.99
        smoothed_normalized_score = smoothed_normalized_score * smoothing + normalized_score * (1.0 - smoothing)
        smoothed_completion = smoothed_completion * smoothing + completion * (1.0 - smoothing)

        # Print logs
        if episode_idx % checkpoint_interval == 0:
            policy.save('checkpoints/checkpoint_' + str(episode_idx) + '.pth')

            if save_replay_buffer:
                policy.save_replay_buffer('replay_buffers/replay_buffer_' + str(episode_idx) + '.pkl')

            if train_params['render']:
                env_renderer.close_window()

        print(
            '\r🚂 Episode {}'
            '\t 🏆 Score: {:.3f}'
            ' Avg: {:.3f}'
            '\t 💯 Done: {:.2f}%'
            ' Avg: {:.2f}%'
            '\t 🎲 Epsilon: {:.3f} '
            '\t 🔀 Action Probs: {}'.format(
                episode_idx,
                normalized_score,
                smoothed_normalized_score,
                100 * completion,
                100 * smoothed_completion,
                eps_start,
                format_action_prob(action_probs)
            ), end=" ")

        # Evaluate policy and log results at some interval
        if (episode_idx % checkpoint_interval == 0 and n_eval_episodes > 0) or episode_idx == n_episodes - 1:
            scores, completions, nb_steps_eval = eval_policy(eval_env, policy, train_params, obs_params)
            print("\t🔍 Evaluation score: {:.3f} done: {:.1f}%".format(np.mean(scores), np.mean(completions) * 100.0))


def format_action_prob(action_probs):
    action_probs = np.round(action_probs, 3)
    actions = ["↻", "←", "↑", "→", "◼"]

    buffer = ""
    for action, action_prob in zip(actions, action_probs):
        buffer += action + " " + "{:.3f}".format(action_prob) + " "

    return buffer


def eval_policy(env, policy, train_params, obs_params):
    n_eval_episodes = train_params['n_eval_episodes']
    tree_depth = obs_params['observation_tree_depth']
    observation_radius = obs_params['observation_radius']

    if train_params.get('LSTM'):
        LSTM = train_params['LSTM']

    scores = []
    completions = []
    nb_steps = []

    for episode_idx in range(n_eval_episodes):

        obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
        
        max_steps = env._max_episode_steps
        action_dict = dict()
        agent_obs = [None] * env.get_num_agents()    

        score = 0.0

        final_step = 0

        for step in range(max_steps):
            for agent in env.get_agent_handles():
                if obs[agent]:
                    if LSTM:
                        obs_list = normalize_cutils(obs, env)
                        agent_obs[agent] = get_features([individual_from_obs_list(obs_list[0], agent)])
                    else:
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

    return scores, completions, nb_steps