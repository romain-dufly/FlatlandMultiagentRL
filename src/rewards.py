from flatland.envs.rail_env import RailEnv
from flatland.envs.step_utils.states import TrainState
from flatland.envs.line_generators import SparseLineGen
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen
from flatland.envs.rail_generators import SparseRailGen
from flatland.envs.observations import TreeObsForRailEnv

import numpy as np

# Custom environment with dense reward from Team JBR_HSE
class RailEnvJBR(RailEnv) :
    """
    Custom environment with dense reward from Team JBR_HSE.

    step_reward = 0.01x∆d-5xis_deadlocked+10xhas_finished,
    where ∆d is the difference between the minimal distances to the destination point from its previous and current positions
    is_deadlocked is 1 if the agent is in a deadlock situation, 0 otherwise
    has_finished is 1 if the agent has finished, 0 otherwise

    Reference : Flatland Competition 2020, Team JBR_HSE : Konstantin Makhnev, Oleg Svidchenko, Vladimir Egorov, Dmitry Ivanov, 
    Aleksei Shpilman; 1st place RL. https://arxiv.org/abs/2103.16511
    """
    def __init__(self, w_deadlocked, w_finished, normalized=False, *args, **kwargs) :
        super().__init__(*args, **kwargs)
        self.w_deadlocked = w_deadlocked
        self.w_finished = w_finished
        self.normalized = normalized
    
    def update_step_rewards(self,i_agent) :
        """
        Update the rewards for the current step
        """

        has_finished = 0
        is_deadlocked = 0

        # Get the current and last position of the agent
        position_curr = self.agents[i_agent].position
        position_last = self.agents[i_agent].old_position
        # Get the current and last direction of the agent
        direction_curr = self.agents[i_agent].direction
        direction_last = self.agents[i_agent].old_direction

        # Difference between minimal distances to the destination point from its previous and current positions
        delta_distance = 0
        if position_curr is not None and position_last is not None :

            # Check if agent took a wrong route and cannot reach the destination
            if self.distance_map.get()[i_agent,position_curr[0],position_curr[1],direction_curr] :
                delta_distance = 0
                is_deadlocked = 1
            else : 
                delta_distance = (self.distance_map.get()[i_agent,position_curr[0],position_curr[1],direction_curr] - 
                               self.distance_map.get()[i_agent,position_last[0],position_last[1],direction_last])
            
        # Check if agent is in a deadlock situation
        # Deadlock situations are marked with a purple color in motionCheck graph
        dAttr = self.motionCheck.G.nodes.get(position_curr)
        if dAttr is None:
            dAttr = {}
        if "color" in dAttr:
            if dAttr["color"] == "purple":
                is_deadlocked = 1

        # Check if agent has finished
        has_finished = 0
        if self.agents[i_agent].state == TrainState.DONE :
            has_finished = 1

        # Update the step reward
        step_reward = 0.01 * delta_distance - self.w_deadlocked * is_deadlocked + self.w_finished * has_finished

        if self.normalized :
            step_reward = self.normalize_reward(step_reward)

        self.rewards_dict[i_agent] += step_reward

    def _handle_end_reward(self,agent) :
        """No specific end reward in this environment"""
        return 0
    
    def normalize_reward(self, reward) :
        """Normalize the reward"""
        return reward/self._max_episode_steps

# Custom environment with dense reward from Team JBR_HSE and environment end reward
class RailEnvJBR_End(RailEnv) :
    """
    Same as RailEnvJBR but with environment end reward as well.
    """
    def __init__(self, w_deadlocked, w_finished, normalized=False, *args, **kwargs) :
        super().__init__(*args, **kwargs)
        self.w_deadlocked = w_deadlocked
        self.w_finished = w_finished
        self.normalized = normalized
    
    def update_step_rewards(self,i_agent) :
        """
        Update the rewards for the current step
        """

        has_finished = 0
        is_deadlocked = 0

        # Get the current and last position of the agent
        position_curr = self.agents[i_agent].position
        position_last = self.agents[i_agent].old_position
        # Get the current and last direction of the agent
        direction_curr = self.agents[i_agent].direction
        direction_last = self.agents[i_agent].old_direction

        # Difference between minimal distances to the destination point from its previous and current positions
        delta_distance = 0
        if position_curr is not None and position_last is not None :

            # Check if agent took a wrong route and cannot reach the destination
            if self.distance_map.get()[i_agent,position_curr[0],position_curr[1],direction_curr] :
                delta_distance = 0
                is_deadlocked = 1
            else : 
                delta_distance = (self.distance_map.get()[i_agent,position_curr[0],position_curr[1],direction_curr] - 
                               self.distance_map.get()[i_agent,position_last[0],position_last[1],direction_last])
            
        # Check if agent is in a deadlock situation
        # Deadlock situations are marked with a purple color in motionCheck graph
        dAttr = self.motionCheck.G.nodes.get(position_curr)
        if dAttr is None:
            dAttr = {}
        if "color" in dAttr:
            if dAttr["color"] == "purple":
                is_deadlocked = 1

        # Check if agent has finished
        has_finished = 0
        if self.agents[i_agent].state == TrainState.DONE :
            has_finished = 1

        # Update the step reward
        step_reward = 0.01 * delta_distance - self.w_deadlocked * is_deadlocked + self.w_finished * has_finished

        if self.normalized :
            step_reward = self.normalize_reward(step_reward)

        self.rewards_dict[i_agent] += step_reward

    def normalize_reward(self, reward) :
        """Normalize the reward"""
        return reward/self._max_episode_steps

class RailEnvDense(RailEnv) :
    """
    Gives the environmental end reward at each timestep.
    """

    def __init__(self, cancellation_factor, collective=False, normalized=False, double_end=False, *args, **kwargs) :
        super().__init__(*args, **kwargs)
        self.cancellation_factor = cancellation_factor
        self.normalized = normalized
        self.collective = collective
        self.double_end = double_end
        self.collective_reward = 0
        self.last_collective_reward_computed = -1

    def reset(self, regenerate_rail=True, regenerate_schedule=True, random_seed=None) :
        observation_dict, info_dict = super().reset(regenerate_rail=regenerate_rail, 
                                        regenerate_schedule=regenerate_schedule, random_seed=random_seed)
        self.collective_reward = 0
        self.last_collective_reward_computed = -1

        return observation_dict, info_dict

    def environment_reward(self, i_agent=-1, agent=None) :
        """Return the implemented end reward for the agent"""
        if agent is None :
            agent = self.agents[i_agent]
        reward = None
        # agent done? (arrival_time is not None)
        if agent.state == TrainState.DONE:
            # if agent arrived earlier or on time = 0
            # if agent arrived later = -ve reward based on how late
            reward = min(agent.latest_arrival - agent.arrival_time, 0)

        # Agents not done (arrival_time is None)
        else:
            # CANCELLED check (never departed)
            if (agent.state.is_off_map_state()):
                reward = -1 * self.cancellation_factor * \
                    (agent.get_travel_time_on_shortest_path(self.distance_map) + self.cancellation_time_buffer)

            # Departed but never reached
            if (agent.state.is_on_map_state()):
                reward = agent.get_current_delay(self._elapsed_steps, self.distance_map)
        
        return reward
    
    def update_step_rewards(self,i_agent) :
        """
        Update the rewards for the current step
        """
        if self.collective :
            if self._elapsed_steps > self.last_collective_reward_computed :
                collective_reward = 0
                for i in range(self.get_num_agents()) :
                    collective_reward += self.environment_reward(i)
                if self.normalized :
                    collective_reward = self.normalize_reward(collective_reward)
                self.last_collective_reward_computed = self._elapsed_steps
                self.collective_reward = collective_reward
                for i in range(self.get_num_agents()) :
                    self.rewards_dict[i] += self.collective_reward

        else :
            step_reward = self.environment_reward(i_agent)
            if self.normalized :
                step_reward = self.normalize_reward(step_reward)

            self.rewards_dict[i_agent] += step_reward

    def _handle_end_reward(self,agent) :
        """No specific end reward in this environment"""

        if self.double_end :
            return self.environment_reward(agent=agent)
        return 0

    def normalize_reward(self, reward) :
        """Normalize the reward"""
        if self.collective :
            return reward/(self._max_episode_steps*self.get_num_agents())
        return reward/(self._max_episode_steps)

    

def get_testing_environements(
    width=20,
    height=15,
    rail_generator=SparseRailGen(
        max_num_cities=2,  # Number of cities
        grid_mode=True,
        max_rails_between_cities=2,
        max_rail_pairs_in_city=1,
    ),
    line_generator=SparseLineGen(speed_ratio_map={1.: 1.}
        ),
    number_of_agents=2, 
    obs_builder_object=TreeObsForRailEnv(max_depth=3),
    malfunction_generator=ParamMalfunctionGen(
        MalfunctionParameters(
            malfunction_rate=0.,  # Rate of malfunction
            min_duration=3,  # Minimal duration
            max_duration=20,  # Max duration
        )
    ),
    ) :

    test_envs = {}
    test_envs['baseline'] = RailEnv(
        width=width,
        height=height,
        rail_generator=rail_generator,
        line_generator=line_generator,
        number_of_agents=number_of_agents,
        obs_builder_object=obs_builder_object,
        malfunction_generator=malfunction_generator
    )

    test_envs['JBR'] = RailEnvJBR(
        w_deadlocked=5,
        w_finished=10,
        width=width,
        height=height,
        rail_generator=rail_generator,
        line_generator=line_generator,
        number_of_agents=number_of_agents,
        obs_builder_object=obs_builder_object,
        malfunction_generator=malfunction_generator
    )

    # Modified weights of JBR
    test_envs['JBR_1'] = RailEnvJBR(
        w_deadlocked=1,
        w_finished=10,
        width=width,
        height=height,
        rail_generator=rail_generator,
        line_generator=line_generator,
        number_of_agents=number_of_agents,
        obs_builder_object=obs_builder_object,
        malfunction_generator=malfunction_generator
    )

    # Modified weights of JBR
    test_envs['JBR_2'] = RailEnvJBR(
        w_deadlocked=1,
        w_finished=20,
        width=width,
        height=height,
        rail_generator=rail_generator,
        line_generator=line_generator,
        number_of_agents=number_of_agents,
        obs_builder_object=obs_builder_object,
        malfunction_generator=malfunction_generator
    )

    # Modified weights of JBR and normalized rewards
    test_envs['JBR_Norm'] = RailEnvJBR(
        w_deadlocked=1,
        w_finished=20,
        normalized=True,
        width=width,
        height=height,
        rail_generator=rail_generator,
        line_generator=line_generator,
        number_of_agents=number_of_agents,
        obs_builder_object=obs_builder_object,
        malfunction_generator=malfunction_generator
    )

    # Step reward from modified JBR and flatland reward
    test_envs['JBR_modified'] = RailEnvJBR_End(
        w_deadlocked=1,
        w_finished=20,
        normalized=True,
        width=width,
        height=height,
        rail_generator=rail_generator,
        line_generator=line_generator,
        number_of_agents=number_of_agents,
        obs_builder_object=obs_builder_object,
        malfunction_generator=malfunction_generator
    )

    # Dense reward from the flatland reward
    test_envs['Dense_base'] = RailEnvDense(
        cancellation_factor=1,
        normalized=True,
        width=width,
        height=height,
        rail_generator=rail_generator,
        line_generator=line_generator,
        number_of_agents=number_of_agents,
        obs_builder_object=obs_builder_object,
        malfunction_generator=malfunction_generator
    )

    # Common step reward for all agents from Dense_base
    test_envs['Collective_base'] = RailEnvDense(
        cancellation_factor=1,
        normalized=True,
        collective=True,
        width=width,
        height=height,
        rail_generator=rail_generator,
        line_generator=line_generator,
        number_of_agents=number_of_agents,
        obs_builder_object=obs_builder_object,
        malfunction_generator=malfunction_generator
    )

    # Dense reward + end reward (so double reward for last step)
    test_envs['DoubleEnd'] = RailEnvDense(
        cancellation_factor=1,
        normalized=True,
        double_end=True,
        width=width,
        height=height,
        rail_generator=rail_generator,
        line_generator=line_generator,
        number_of_agents=number_of_agents,
        obs_builder_object=obs_builder_object,
        malfunction_generator=malfunction_generator
    )

    for env in test_envs.values() :
        _,_ = env.reset()

    return test_envs