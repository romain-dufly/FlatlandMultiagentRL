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
    def __init__(self, *args, **kwargs) :
        super().__init__(*args, **kwargs)
    
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
        step_reward = 0.01 * delta_distance - 5 * is_deadlocked + 10 * has_finished

        self.rewards_dict[i_agent] += step_reward

    def _handle_end_reward(self,agent) :
        """No specific end reward in this environment"""
        return 0
    

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