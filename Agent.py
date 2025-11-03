from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np

from strategy.Assignment import role_assignment 
from strategy.Strategy import Strategy 

from formation.Formation import GenerateBasicFormation


class Agent(Base_Agent):
    def __init__(self, host:str, agent_port:int, monitor_port:int, unum:int,
                 team_name:str, enable_log, enable_draw, wait_for_server=True, is_fat_proxy=False) -> None:
        
        # define robot type
        robot_type = (0,1,1,1,2,3,3,3,4,4,4)[unum-1]

        # Initialize base agent
        # Args: Server IP, Agent Port, Monitor Port, Uniform No., Robot Type, Team Name, Enable Log, Enable Draw, play mode correction, Wait for Server, Hear Callback
        super().__init__(host, agent_port, monitor_port, unum, robot_type, team_name, enable_log, enable_draw, True, wait_for_server, None)

        self.enable_draw = enable_draw
        self.state = 0  # 0-Normal, 1-Getting up, 2-Kicking
        self.kick_direction = 0
        self.kick_distance = 0
        self.fat_proxy_cmd = "" if is_fat_proxy else None
        self.fat_proxy_walk = np.zeros(3) # filtered walk parameters for fat proxy

        self.init_pos = ([-14,0],[-9,-5],[-9,0],[-9,5],[-5,-5],[-5,0],[-5,5],[-1,-6],[-1,-2.5],[-1,2.5],[-1,6])[unum-1] # initial formation


    def beam(self, avoid_center_circle=False):
        r = self.world.robot
        pos = self.init_pos[:] # copy position list 
        self.state = 0

        # Avoid center circle by moving the player back 
        if avoid_center_circle and np.linalg.norm(self.init_pos) < 2.5:
            pos[0] = -2.3 

        if np.linalg.norm(pos - r.loc_head_position[:2]) > 0.1 or self.behavior.is_ready("Get_Up"):
            self.scom.commit_beam(pos, M.vector_angle((-pos[0],-pos[1]))) # beam to initial position, face coordinate (0,0)
        else:
            if self.fat_proxy_cmd is None: # normal behavior
                self.behavior.execute("Zero_Bent_Knees_Auto_Head")
            else: # fat proxy behavior
                self.fat_proxy_cmd += "(proxy dash 0 0 0)"
                self.fat_proxy_walk = np.zeros(3) # reset fat proxy walk


    def move(self, target_2d=(0,0), orientation=None, is_orientation_absolute=True,
             avoid_obstacles=True, priority_unums=[], is_aggressive=False, timeout=3000):
        '''
        Walk to target position

        Parameters
        ----------
        target_2d : array_like
            2D target in absolute coordinates
        orientation : float
            absolute or relative orientation of torso, in degrees
            set to None to go towards the target (is_orientation_absolute is ignored)
        is_orientation_absolute : bool
            True if orientation is relative to the field, False if relative to the robot's torso
        avoid_obstacles : bool
            True to avoid obstacles using path planning (maybe reduce timeout arg if this function is called multiple times per simulation cycle)
        priority_unums : list
            list of teammates to avoid (since their role is more important)
        is_aggressive : bool
            if True, safety margins are reduced for opponents
        timeout : float
            restrict path planning to a maximum duration (in microseconds)    
        '''
        r = self.world.robot

        if self.fat_proxy_cmd is not None: # fat proxy behavior
            self.fat_proxy_move(target_2d, orientation, is_orientation_absolute) # ignore obstacles
            return

        if avoid_obstacles:
            target_2d, _, distance_to_final_target = self.path_manager.get_path_to_target(
                target_2d, priority_unums=priority_unums, is_aggressive=is_aggressive, timeout=timeout)
        else:
            distance_to_final_target = np.linalg.norm(target_2d - r.loc_head_position[:2])

        self.behavior.execute("Walk", target_2d, True, orientation, is_orientation_absolute, distance_to_final_target) # Args: target, is_target_abs, ori, is_ori_abs, distance


    def kick(self, kick_direction=None, kick_distance=None, abort=False, enable_pass_command=False):
        '''
        Walk to ball and kick

        Parameters
        ----------
        kick_direction : float
            kick direction, in degrees, relative to the field
        kick_distance : float
            kick distance in meters
        abort : bool
            True to abort.
            The method returns True upon successful abortion, which is immediate while the robot is aligning itself. 
            However, if the abortion is requested during the kick, it is delayed until the kick is completed.
        avoid_pass_command : bool
            When False, the pass command will be used when at least one opponent is near the ball
            
        Returns
        -------
        finished : bool
            Returns True if the behavior finished or was successfully aborted.
        '''
        return self.behavior.execute("Dribble",None,None)

        if self.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None: # normal behavior
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort) # Basic_Kick has no kick distance control
        else: # fat proxy behavior
            return self.fat_proxy_kick()


    def kickTarget(self, strategyData, mypos_2d=(0,0),target_2d=(0,0), abort=False, enable_pass_command=False):
        '''
        Walk to ball and kick

        Parameters
        ----------
        kick_direction : float
            kick direction, in degrees, relative to the field
        kick_distance : float
            kick distance in meters
        abort : bool
            True to abort.
            The method returns True upon successful abortion, which is immediate while the robot is aligning itself. 
            However, if the abortion is requested during the kick, it is delayed until the kick is completed.
        avoid_pass_command : bool
            When False, the pass command will be used when at least one opponent is near the ball
            
        Returns
        -------
        finished : bool
            Returns True if the behavior finished or was successfully aborted.
        '''

        # Calculate the vector from the current position to the target position
        vector_to_target = np.array(target_2d) - np.array(mypos_2d)
        
        # Calculate the distance (magnitude of the vector)
        kick_distance = np.linalg.norm(vector_to_target)
        
        # Calculate the direction (angle) in radians
        direction_radians = np.arctan2(vector_to_target[1], vector_to_target[0])
        
        # Convert direction to degrees for easier interpretation (optional)
        kick_direction = np.degrees(direction_radians)


        if strategyData.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None: # normal behavior
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort) # Basic_Kick has no kick distance control
        else: # fat proxy behavior
            return self.fat_proxy_kick()

    def think_and_send(self):
        
        behavior = self.behavior
        strategyData = Strategy(self.world)
        d = self.world.draw

        if strategyData.play_mode == self.world.M_GAME_OVER:
            pass
        elif strategyData.PM_GROUP == self.world.MG_ACTIVE_BEAM:
            self.beam()
        elif strategyData.PM_GROUP == self.world.MG_PASSIVE_BEAM:
            self.beam(True) # avoid center circle
        elif self.state == 1 or (behavior.is_ready("Get_Up") and self.fat_proxy_cmd is None):
            self.state = 0 if behavior.execute("Get_Up") else 1
        else:
            if strategyData.play_mode != self.world.M_BEFORE_KICKOFF:
                self.select_skill(strategyData)
            else:
                pass


        #--------------------------------------- 3. Broadcast
        self.radio.broadcast()

        #--------------------------------------- 4. Send to server
        if self.fat_proxy_cmd is None: # normal behavior
            self.scom.commit_and_send( strategyData.robot_model.get_command() )
        else: # fat proxy behavior
            self.scom.commit_and_send( self.fat_proxy_cmd.encode() ) 
            self.fat_proxy_cmd = ""


    ##theto
    # here i check if i am the closest player to the ball
    def active_player_score(self, strategyData):
        """
        Score yourself for being the active player.
        Higher score = more suitable to go for the ball.
        """
        my_pos = np.array(strategyData.mypos)
        ball_pos = np.array(strategyData.ball_2d)
        
        # distance to ball (closer = better)
        dist_to_ball = np.linalg.norm(my_pos - ball_pos)
        
        # distance from ball to goal (closer = more urgent)
        goal_pos = np.array([15, 0])
        ball_to_goal = np.linalg.norm(goal_pos - ball_pos)
        
        # closest opponent distance
        opponent_dist = min([np.linalg.norm(np.array(op) - ball_pos) 
                            for op in strategyData.opponent_positions if op is not None] + [0])
        
        # score formula
        score = -dist_to_ball + 0.5*(15 - ball_to_goal) + 0.5*opponent_dist
        return score

    def am_i_the_active_player(self, strategyData):
       
        scores = []
        for i, teammate_pos in enumerate(strategyData.teammate_positions):
            if teammate_pos is None:
                scores.append(-1000)
                continue
            temp_data = strategyData
            temp_data.mypos = teammate_pos
            scores.append(self.active_player_score(temp_data))
        
        best_player_num = np.argmax(scores) + 1
        return strategyData.player_unum == best_player_num


    def can_kick_safely(self, strategyData, target):
        
        ball_pos = np.array(strategyData.ball_2d)
        target = np.array(target)
        vec_to_target = target - ball_pos
        
        for op in strategyData.opponent_positions:
            if op is None:
                continue
            op_vec = np.array(op) - ball_pos
            proj = np.dot(op_vec, vec_to_target) / (np.linalg.norm(vec_to_target)**2)
            if 0 < proj < 1:
                closest_point = ball_pos + proj * vec_to_target
                if np.linalg.norm(closest_point - np.array(op)) < 0.8:
                    return False
        return True

    #theto (here i check if i am close enough to kick the ball)
    def am_i_close_enough_to_kick(self, strategyData):
        my_position = strategyData.mypos
        ball_position = strategyData.ball_2d

        distance_from_ball = np.linalg.norm(my_position - ball_position)

        if distance_from_ball < 10.0:
            return True
        else:
            return False

    #theto (here i am deciding if i should kick)
    def should_i_kick(self, strategyData):
        current_play_mode = strategyData.play_mode
        
        if current_play_mode != self.world.M_PLAY_ON:
            return False
        elif not self.am_i_the_active_player(strategyData):
            return False
        else:
            return True

    #theto (here i handle other game modes)
    def handle_game_mode(self, strategyData):
        current_mode = strategyData.play_mode
        my_player_number = strategyData.player_unum

        if current_mode == self.world.M_BEFORE_KICKOFF:
            return "move"

        elif current_mode == self.world.M_OUR_KICKOFF:
            return "kick" if my_player_number == 1 else "move"

        elif current_mode == self.world.M_THEIR_KICKOFF:
            return "move"

        elif current_mode in [self.world.M_OUR_KICK_IN,
                                self.world.M_OUR_CORNER_KICK,
                                self.world.M_OUR_FREE_KICK]:
           
            return "kick" if strategyData.active_player_unum == my_player_number else "move"

        elif current_mode in [self.world.M_THEIR_KICK_IN,
                                self.world.M_THEIR_CORNER_KICK,
                                self.world.M_THEIR_FREE_KICK]:
            
            return "move"

        elif current_mode == self.world.M_OUR_GOAL_KICK:
            return "kick" if my_player_number == 1 else "move"

        elif current_mode == self.world.M_THEIR_GOAL_KICK:
            return "move"

        elif current_mode == self.world.M_OUR_PENALTY:
            return "kick" if my_player_number == 1 else "move"

        elif current_mode == self.world.M_THEIR_PENALTY:
            return "move"

        elif current_mode == self.world.M_PLAY_ON:
            return "normal"

        elif current_mode == self.world.M_GAME_OVER:
            return "none"

        else:
            return "none"



    
    def select_skill(self, strategyData):
        drawer = self.world.draw
        mode_action = self.handle_game_mode(strategyData)

        if mode_action == "kick":
            target = strategyData.kick_Target(strategyData)
            drawer.annotation((0, 0), "KICK MODE", drawer.Color.yellow, "status")
            return self.kickTarget(strategyData, strategyData.mypos, target)

        elif mode_action == "move":
            target = strategyData.move_Target(strategyData)
            drawer.annotation((0, 0), "MOVE MODE", drawer.Color.blue, "status")
            return self.move(target, orientation=strategyData.ball_dir)

        elif mode_action == "normal":
            # Regular play
            if self.am_i_the_active_player(strategyData) and self.am_i_close_enough_to_kick(strategyData):
                target = strategyData.kick_Target(strategyData)
                drawer.annotation((0, 0), "PLAY ON: KICKING", drawer.Color.green, "status")
                return self.kickTarget(strategyData, strategyData.mypos, target)
            else:
                target = strategyData.move_Target(strategyData)
                drawer.annotation((0, 0), "PLAY ON: MOVING", drawer.Color.blue, "status")
                return self.move(target, orientation=strategyData.ball_dir)

        elif mode_action == "none":
            drawer.annotation((0, 0), "IDLE", drawer.Color.white, "status")
            return None


    #--------------------------------------- Fat proxy auxiliary methods

    def fat_proxy_kick(self):
        w = self.world
        r = self.world.robot 
        ball_2d = w.ball_abs_pos[:2]
        my_head_pos_2d = r.loc_head_position[:2]

        if np.linalg.norm(ball_2d - my_head_pos_2d) < 0.25:
            # fat proxy kick arguments: power [0,10]; relative horizontal angle [-180,180]; vertical angle [0,70]
            self.fat_proxy_cmd += f"(proxy kick 10 {M.normalize_deg( self.kick_direction  - r.imu_torso_orientation ):.2f} 20)" 
            self.fat_proxy_walk = np.zeros(3) # reset fat proxy walk
            return True
        else:
            self.fat_proxy_move(ball_2d-(-0.1,0), None, True) # ignore obstacles
            return False


    def fat_proxy_move(self, target_2d, orientation, is_orientation_absolute):
        r = self.world.robot

        target_dist = np.linalg.norm(target_2d - r.loc_head_position[:2])
        target_dir = M.target_rel_angle(r.loc_head_position[:2], r.imu_torso_orientation, target_2d)

        if target_dist > 0.1 and abs(target_dir) < 8:
            self.fat_proxy_cmd += (f"(proxy dash {100} {0} {0})")
            return

        if target_dist < 0.1:
            if is_orientation_absolute:
                orientation = M.normalize_deg( orientation - r.imu_torso_orientation )
            target_dir = np.clip(orientation, -60, 60)
            self.fat_proxy_cmd += (f"(proxy dash {0} {0} {target_dir:.1f})")
        else:
            self.fat_proxy_cmd += (f"(proxy dash {20} {0} {target_dir:.1f})")