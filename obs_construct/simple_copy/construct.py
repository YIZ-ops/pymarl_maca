import numpy as np


class ObsConstruct:
    def __init__(self, size_x, size_y, detector_num, fighter_num):
        self.battlefield_size_x = size_x
        self.battlefield_size_y = size_y
        self.detector_num = detector_num
        self.fighter_num = fighter_num

    def obs_construct(self, teammate_obs_raw_dict, enemy_detector_num, enemy_fighter_num):
        teammate_detector_data_obs_list = teammate_obs_raw_dict["detector_obs_list"]
        teammate_fighter_data_obs_list = teammate_obs_raw_dict["fighter_obs_list"]
        teammate_joint_data_obs_dict = teammate_obs_raw_dict["joint_obs_dict"]

        data_obs = self.__get_data_obs(
            teammate_detector_data_obs_list,
            teammate_fighter_data_obs_list,
            enemy_detector_num,
            enemy_fighter_num,
            teammate_joint_data_obs_dict,
        )
        return data_obs 

    def __get_data_obs(
        self,
        teammate_detector_data_obs_list,
        teammate_fighter_data_obs_list,
        enemy_detector_data_obs_list,
        enemy_fighter_data_obs_list,
        teammate_joint_data_obs_dict
    ):

        data_obs = self.get_agent_observation_data(
            teammate_detector_data_obs_list,
            teammate_fighter_data_obs_list,
            enemy_detector_data_obs_list,
            enemy_fighter_data_obs_list,
            teammate_joint_data_obs_dict
        )
        return data_obs

    def get_agent_observation_data(self, teammate_detector_data, teammate_fighter_data, enemy_detector_count, enemy_fighter_count, teammate_joint_data_obs_dict):
        teammate_detector_count = len(teammate_detector_data)
        teammate_fighter_count = len(teammate_fighter_data)
        
        data_obs = np.zeros((teammate_detector_count + teammate_fighter_count, 
                             teammate_detector_count + teammate_fighter_count + enemy_detector_count + enemy_fighter_count, 
                             7))
        
        for i, data in enumerate(teammate_detector_data + teammate_fighter_data):
            # 自己
            if data["alive"]:
                data_obs[i, 0, :] = [
                    data["id"],
                    1 if data.get("l_missile_left", 0) == 0 else 2,
                    data["pos_x"],
                    data["pos_y"],
                    data["course"],
                    data.get("l_missile_left", 0),
                    data.get("s_missile_left", 0),
                ]
                
            # 队友 detector
            detector_list = [detector for detector in teammate_detector_data if detector["id"] != data["id"]]
            for index, detector in enumerate(detector_list):
                if detector["alive"]:
                    data_obs[i, 1 + index, :] = [
                        detector["id"],
                        1,
                        detector["pos_x"],
                        detector["pos_y"],
                        detector["course"],
                        0,
                        0,
                    ]
                
            # 队友 fighter
            fighter_list = [fighter for fighter in teammate_fighter_data if fighter["id"] != data["id"]]
            for index, fighter in enumerate(fighter_list):
                if fighter["alive"]:
                    data_obs[i, len(detector_list) + 1 + index, :] = [
                        fighter["id"],
                        2,
                        fighter["pos_x"],
                        fighter["pos_y"],
                        fighter["course"],
                        fighter["l_missile_left"],
                        fighter["s_missile_left"],
                    ]

            # 敌人 detector
            visible_detector_list = [enemy for enemy in data["r_visible_list"] if enemy["type"] == 0]
            passive_visible_detector_list = [enemy for enemy in teammate_joint_data_obs_dict["passive_detection_enemy_list"] if enemy["type"] == 0] 
            all_detector_list = visible_detector_list + passive_visible_detector_list
            unique_detector_list = []
            for enemy in all_detector_list:
                if enemy not in unique_detector_list:
                    unique_detector_list.append(enemy)

            for index, detector in enumerate(unique_detector_list):
                data_obs[i, teammate_detector_count + teammate_fighter_count + index, :] = [
                    detector["id"],
                    detector["type"],
                    detector["pos_x"],
                    detector["pos_y"],
                    0,
                    0,
                    0,
                ]
        
            # 敌人 fighter
            visible_fighter_list = [enemy for enemy in data["r_visible_list"] if enemy["type"] == 1]
            passive_visible_fighter_list = [enemy for enemy in teammate_joint_data_obs_dict["passive_detection_enemy_list"] if enemy["type"] == 1]
            all_fighter_list = visible_fighter_list + passive_visible_fighter_list
            unique_fighter_list = []
            for enemy in all_fighter_list:
                if enemy not in unique_fighter_list:
                    unique_fighter_list.append(enemy)

            for index, fighter in enumerate(unique_fighter_list):
                data_obs[i, teammate_detector_count + teammate_fighter_count + enemy_detector_count + index, :] = [
                    fighter["id"],
                    fighter["type"],
                    fighter["pos_x"],
                    fighter["pos_y"],
                    0,
                    0,
                    0,
                ]

        total_columns = (teammate_detector_count + teammate_fighter_count + enemy_detector_count + enemy_fighter_count) * 7
        data_obs_2d = data_obs.reshape((teammate_detector_count + teammate_fighter_count, total_columns))
        return data_obs_2d
    

class StateConstruct:
    def __init__(self, size_x, size_y, teammate_detector_num, teammate_fighter_num, enemy_detector_num, enemy_figher_num):
        self.battlefield_size_x = size_x
        self.battlefield_size_y = size_y
        self.teammate_detector_num = teammate_detector_num
        self.temmate_fighter_num = teammate_fighter_num
        self.enemy_detector_num = enemy_detector_num
        self.enemy_figher_num = enemy_figher_num

    def state_construct(self, teammate_obs_raw_dict, enemy_obs_raw_dict):
        teammate_detector_data_obs_list = teammate_obs_raw_dict["detector_obs_list"]
        teammate_fighter_data_obs_list = teammate_obs_raw_dict["fighter_obs_list"]
        enemy_detector_data_obs_list = enemy_obs_raw_dict["detector_obs_list"]
        enemy_fighter_data_obs_list = enemy_obs_raw_dict["fighter_obs_list"]

        data_obs = self.__get_data_obs(
            teammate_detector_data_obs_list,
            teammate_fighter_data_obs_list,
            enemy_detector_data_obs_list,
            enemy_fighter_data_obs_list,
        )
        return data_obs 

    def __get_data_obs(
        self,
        teammate_detector_data_obs_list,
        teammate_fighter_data_obs_list,
        enemy_detector_data_obs_list,
        enemy_fighter_data_obs_list,
    ):

        data_obs = self.get_agent_state_data(
            teammate_detector_data_obs_list,
            teammate_fighter_data_obs_list,
            enemy_detector_data_obs_list,
            enemy_fighter_data_obs_list,
        )
        return data_obs

    def get_agent_state_data(self, teammate_detector_data, teammate_fighter_data, enemy_detector_data, enemy_fighter_data):
        teammate_detector_count = len(teammate_detector_data)
        teammate_fighter_count = len(teammate_fighter_data)
        enemy_detector_count = len(enemy_detector_data)
        enemy_fighter_count = len(enemy_fighter_data)
        
        data_obs = np.zeros((teammate_detector_count + teammate_fighter_count + enemy_detector_count + enemy_fighter_count, 
                             7))
        
        for i, data in enumerate(teammate_detector_data + teammate_fighter_data +  enemy_detector_data + enemy_fighter_data):
            if data["alive"]:
                data_obs[i, :] = [
                    data["id"],
                    1 if data.get("l_missile_left", 0) == 0 else 2,
                    data["pos_x"],
                    data["pos_y"],
                    data["course"],
                    data.get("l_missile_left", 0),
                    data.get("s_missile_left", 0),
                ]
        total_columns = (teammate_detector_count + teammate_fighter_count + enemy_detector_count + enemy_fighter_count) * 7
        data_state_1d = data_obs.reshape((total_columns))
        return data_state_1d