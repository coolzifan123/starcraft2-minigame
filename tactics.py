import numpy as np
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib import units

FUNCTIONS = actions.FUNCTIONS


# class tactic
# contains 3 functions:
#   check_available_func
#   select_executer_func
#   exec_func

SCALE=32/84
class Tactic:
    def __init__(self, check_available_func, select_executer_func, exec_func, once=False):
        self.func1 = check_available_func
        self.func2 = select_executer_func
        self.func3 = exec_func
        self.add_func1 = lambda *argv: True
        self.once = once
        self.execed = False

    def check_tactic_executable(self, *argv):
        if self.once:
            if self.execed:
                return False
            else:
                self.execed = True
        return self.func1(*argv) and self.add_func1(*argv)

    def select_executer_func(self, *argv):
        return self.func2(*argv)

    def exec_func(self, *argv):
        return self.func3(*argv)

    def add_additional_check_tactic_executable(self, additional_func):
        self.add_func1 = additional_func


def get_unit_cnt(obs, unit_type):
    cnt = 0
    for unit in obs.observation.feature_units:
        if unit.unit_type == unit_type:
            cnt += 1
    return cnt


def get_one_idle_scv(obs):
    """
    check if any idle scv are available
    :return idle_scv_pos: [x, y], if no, return None
    """
    for unit in obs.observation.feature_units:
        if unit.unit_type == units.Terran.SCV and int(unit.order_length) == 0:
            return [unit.x, unit.y]
    return None


def get_one_idle_marine(obs):
    """
    check if any idle marine are available
    :return idle_scv_pos: [x, y], if no, return None
    """
    for unit in obs.observation.feature_units:
        if unit.unit_type == units.Terran.Marine and int(unit.order_length) == 0:
            return [unit.x, unit.y]
    return None


def get_busy_marine_cnt(obs):
    cnt = 0
    for unit in obs.observation.feature_units:
        if unit.unit_type == units.Terran.Marine and int(unit.order_length) != 0:
            cnt += 1
    return cnt


def get_idle_marine_cnt(obs):
    cnt = 0
    for unit in obs.observation.feature_units:
        if unit.unit_type == units.Terran.Marine and int(unit.order_length) == 0:
            cnt += 1
    return cnt


def get_one_random_scv(obs):
    a = get_one_idle_scv(obs)
    if a:
        return a
    else:
        for unit in obs.observation.feature_units:
            if unit.unit_type == units.Terran.SCV:
                return [unit.x, unit.y]


def get_minerals_positions(obs):
    res = []
    for unit in obs.observation.feature_units:
        if unit.unit_type == units.Neutral.MineralField:
            return [unit.x, unit.y]


def get_mineralshards_positions(obs):
    pos = get_one_idle_marine(obs)
    res = []
    for unit in obs.observation.feature_units:
        if unit.alliance == features.PlayerRelative.NEUTRAL:
            res.append([unit.x, unit.y])
    distances = np.linalg.norm(np.array(res) - np.array(pos), axis=1)
    closest_mineral_xy = res[np.argmin(distances)]
    return closest_mineral_xy


# get command center with fewer orders queued
# command center is in [33, 33], radius = 9 for BuildMarine and CollectMineralsAndGas


def get_command_center_positions(obs):
    order_len = 999
    for unit in obs.observation.feature_units:
        if unit.unit_type == units.Terran.CommandCenter and unit.order_length < order_len:
            pos = [unit.x, unit.y]
            r = unit.radius
            order_len = unit.order_length
    return pos


# get new command center position


def get_new_command_center_position(obs):
    x, y, r = None, None, None
    for unit in obs.observation.feature_units:
        if unit.unit_type == units.Terran.CommandCenter:
            x, y = unit.x, unit.y
            r = unit.radius
    if x < int(42*SCALE):
        x = int(x + 2 *SCALE* r)
    else:
        x = int(x - 2 *SCALE* r)
    return [x, y]


# radius = 6


def get_barracks_position(obs):
    order_len = 999
    for unit in obs.observation.feature_units:
        if unit.unit_type == units.Terran.Barracks and unit.order_length < order_len:
            pos = [unit.x, unit.y]
            order_len = unit.order_length
    return pos


def get_enemy_pos(obs):
    min_health = 99999
    pos = [0, 0]
    for unit in obs.observation.feature_units:
        if unit.alliance == features.PlayerRelative.ENEMY and unit.health < min_health:
            pos = [unit.x, unit.y]
            min_health = unit.health
    return pos


def get_enemy_pos_y_min_max(obs):
    y_min = 888
    y_max = 0
    for unit in obs.observation.feature_units:
        if unit.alliance == features.PlayerRelative.ENEMY:
            if unit.y < y_min:
                y_min = unit.y
                min_pos = [unit.x, unit.y - int(20*SCALE)]
            if unit.y > y_max:
                y_max = unit.y
                max_pos = [unit.x, unit.y + int(20*SCALE)]
    return [min_pos, max_pos]


# get potential supply depot pos


def get_potential_supply_depot_pos(obs):
    potential_barracks_pos = []
    for x in [int(22*SCALE), int(30*SCALE), int(38*SCALE), int(46*SCALE)]:
        for y in [int(12*SCALE), int(20*SCALE), int(46*SCALE), int(54*SCALE)]:
            potential_barracks_pos.append([int(x), int(y)])
    idx = get_unit_cnt(obs, units.Terran.SupplyDepot)
    if idx > len(potential_barracks_pos) - 1:
        for unit in obs.observation.feature_units:
            if unit.unit_type == units.Terran.CommandCenter:
                x, y = unit.x, unit.y
                r = unit.radius
                delta_x = np.random.randint(-int(4*SCALE* r), int(4*SCALE*r))
                delta_y = np.random.randint(-int(4*SCALE * r), int(4*SCALE * r))
                return [x + delta_x, y + delta_y]
    else:
        return potential_barracks_pos[idx]


def get_potential_barracks_pos(obs):
    # for unit in obs.observation.feature_units:
    #     if unit.unit_type == units.Terran.CommandCenter:
    #         x, y = unit.x, unit.y
    #         r = unit.radius
    #         delta_x = np.random.randint(r, 5 * r)
    #         delta_y = np.random.randint(-3 * r, 3 * r)
    #         return [x + delta_x, y + delta_y]

    potential_barracks_pos = []
    for x in [int(56*SCALE), int(69*SCALE), int(72*SCALE)]:
        for y in [int(7*SCALE), int(22*SCALE), int(37*SCALE), int(52*SCALE)]:
            potential_barracks_pos.append([int(x), int(y)])
    a=potential_barracks_pos[get_unit_cnt(obs, units.Terran.Barracks)]
    a[1]+=1
    return a
    # return [0, 0]


def get_zerg_pos(obs):
    min_health = 99999
    pos = [0, 0]
    for unit in obs.observation.feature_units:
        if unit.unit_type == units.Zerg.Zergling:
            pos = [unit.x, unit.y]
            min_health = unit.health
    return pos


def get_baneling_pos(obs):
    pos = [0, 0]
    for unit in obs.observation.feature_units:
        if unit.unit_type == units.Zerg.Baneling:
            pos = [unit.x, unit.y]
            return pos


def food_cap_equal_used(obs):
    return obs.observation.player[features.Player.food_cap] == obs.observation.player[features.Player.food_used]


def food_cap_available(obs):
    return obs.observation.player[features.Player.food_cap] > obs.observation.player[features.Player.food_used]


def make_save_pos(obs, pos):
    x, y = pos[0], pos[1]
    x = min(max(0, x), obs.observation.feature_screen.shape[1] - 1)
    y = min(max(0, y), obs.observation.feature_screen.shape[2] - 1)
    return [x, y]


# build new command center
tactic_build_command_center = Tactic(
    lambda obs, *argv: obs.observation.player[features.Player.minerals] >= 450,
    # FUNCTIONS.select_point("select", make_save_pos(obs, get_one_random_scv(obs))),
    lambda obs, *argv: (2, [[0], [make_save_pos(obs, get_one_random_scv(obs))
                                  [0], make_save_pos(obs, get_one_random_scv(obs))[1]]]),
    lambda obs, *argv: (44, [[0], [make_save_pos(obs, get_new_command_center_position(obs))[0],
                                   make_save_pos(obs, get_new_command_center_position(obs))[1]]]
                        ) if FUNCTIONS.Build_CommandCenter_screen.id in obs.observation.available_actions else (0, [])
)

# build supply depot
tactic_build_supply_depot = Tactic(
    lambda obs, *argv: obs.observation.player[features.Player.minerals] >= 100,
    lambda obs, *argv: (2, [[0], [make_save_pos(obs, get_one_random_scv(obs))[0],
                                  make_save_pos(obs, get_one_random_scv(obs))[1]]]),
    lambda obs, *argv: (91, [[0], [make_save_pos(obs, get_potential_supply_depot_pos(obs))[0],
                                   make_save_pos(obs, get_potential_supply_depot_pos(obs))[1]]]
                        ) if FUNCTIONS.Build_SupplyDepot_screen.id in obs.observation.available_actions else (0, [])
)

# tactic build barracks
tactic_build_barracks = Tactic(
    lambda obs, *argv: obs.observation.player[features.Player.minerals] >= 150,
    lambda obs, *argv: (
        2, [[0], [make_save_pos(obs, get_one_random_scv(obs))[0], make_save_pos(obs, get_one_random_scv(obs))[1]]]),
    lambda obs, *argv:
    (42, [[0], [make_save_pos(obs, get_potential_barracks_pos(obs))[0],
                make_save_pos(obs, get_potential_barracks_pos(obs))[1]]])
    if FUNCTIONS.Build_Barracks_screen.id in obs.observation.available_actions else (0, [])
)

# harvest mineral
tactic_harvest_mineral = Tactic(
    lambda obs, *argv: get_one_idle_scv(obs),
    lambda obs, *argv: (
        2, [[0], [make_save_pos(obs, get_one_idle_scv(obs))[0], make_save_pos(obs, get_one_idle_scv(obs))[1]]]),
    lambda obs, *argv: (264, [[0], [make_save_pos(obs, get_minerals_positions(obs))[0],
                                    make_save_pos(obs, get_minerals_positions(obs))[1]]])
    if FUNCTIONS.Harvest_Gather_screen.id in obs.observation.available_actions else (0, [])
)

# train scv
tactic_train_scv = Tactic(
    lambda obs, *argv: food_cap_available(obs) and
                       get_unit_cnt(obs, units.Terran.CommandCenter) >= 1 and
                       obs.observation.player[features.Player.minerals] >= 50,
    lambda obs, *argv: (2, [[0], [make_save_pos(obs, get_command_center_positions(obs))[0],
                                  make_save_pos(obs, get_command_center_positions(obs))[1]]]),
    lambda obs, *argv: (490, [[0]])
    if FUNCTIONS.Train_SCV_quick.id in obs.observation.available_actions else (0, [])
)

tactic_train_marine = Tactic(
    lambda obs, *argv: food_cap_available(obs) and
                       get_unit_cnt(obs, units.Terran.Barracks) >= 1 and
                       obs.observation.player[features.Player.minerals] >= 50,
    lambda obs, *argv: (2,[[0], [make_save_pos(obs, get_barracks_position(obs))[0],
                                make_save_pos(obs, get_barracks_position(obs))[1]]]),
    lambda obs, *argv: (477, [[0]])
    if FUNCTIONS.Train_Marine_quick.id in obs.observation.available_actions else (0, [])
)

# collect mineral shards
tactic_collect_mineralshards = Tactic(
    lambda obs, *argv: get_one_idle_marine(obs),
    lambda obs, *argv: (
        2, [[0], [make_save_pos(obs, get_one_idle_marine(obs))[0], make_save_pos(obs, get_one_idle_marine(obs))[1]]]),
    lambda obs, *argv:
    (331, [[0], [make_save_pos(obs, get_mineralshards_positions(obs))[0],
                 make_save_pos(obs, get_mineralshards_positions(obs))[1]]])
    if FUNCTIONS.Move_screen.id in obs.observation.available_actions else (0, [])
)

tactic_attack_zerg_pioneer = Tactic(
    lambda obs, *argv: get_one_idle_marine(obs),
    lambda obs, *argv: (
        2, [[0], [make_save_pos(obs, get_one_idle_marine(obs))[0], make_save_pos(obs, get_one_idle_marine(obs))[1]]]),
    lambda obs, *argv: (12, [[0], [make_save_pos(obs, get_zerg_pos(obs))[0], make_save_pos(obs, get_zerg_pos(obs))[1]]])
    if FUNCTIONS.Attack_screen.id in obs.observation.available_actions else (0, [])
)

tactic_attack_baneling_pioneer = Tactic(
    lambda obs, *argv: get_one_idle_marine(obs),
    lambda obs, *argv: (
        2, [[0], [make_save_pos(obs, get_one_idle_marine(obs))[0], make_save_pos(obs, get_one_idle_marine(obs))[1]]]),
    lambda obs, *argv: (
    12, [[0], [make_save_pos(obs, get_baneling_pos(obs))[0], make_save_pos(obs, get_baneling_pos(obs))[1]]])
    if FUNCTIONS.Attack_screen.id in obs.observation.available_actions else (0, [])
)

tactic_attack_all = Tactic(
    lambda obs, *argv: True,
    lambda obs, *argv: (7, [[0]]),
    lambda obs, *argv: (12, [[0], [make_save_pos(obs, get_enemy_pos_y_min_max(obs)[0])[0],
                                   make_save_pos(obs, get_enemy_pos_y_min_max(obs)[0])[1]]])
    if FUNCTIONS.Attack_screen.id in obs.observation.available_actions else (0, [])
)

tactic_attack_zerg_all = Tactic(
    lambda obs, *argv: get_one_idle_marine(obs),
    lambda obs, *argv: (
        2, [[0], [make_save_pos(obs, get_one_idle_marine(obs))[0], make_save_pos(obs, get_one_idle_marine(obs))[1]]]),
    lambda obs, *argv: (12, [[0], [make_save_pos(obs, get_zerg_pos(obs))[0], make_save_pos(obs, get_zerg_pos(obs))[1]]])
    if FUNCTIONS.Attack_screen.id in obs.observation.available_actions else (0, [])
)

tactic_move_to_enemy_top = Tactic(
    lambda obs, *argv: get_one_idle_marine(obs),
    lambda obs, *argv: (
        2, [[0], [make_save_pos(obs, get_one_idle_marine(obs))[0], make_save_pos(obs, get_one_idle_marine(obs))[1]]]),
    lambda obs, *argv: (331, [[0], [make_save_pos(obs, get_enemy_pos_y_min_max(obs)[0])[0],
                                    make_save_pos(obs, get_enemy_pos_y_min_max(obs)[0])[1]]])
    if FUNCTIONS.Attack_screen.id in obs.observation.available_actions else (0, [])
)

tactic_move_to_enemy_bottom = Tactic(
    lambda obs, *argv: get_one_idle_marine(obs),
    lambda obs, *argv: (
        2, [[0], [make_save_pos(obs, get_one_idle_marine(obs))[0], make_save_pos(obs, get_one_idle_marine(obs))[1]]]),
    lambda obs, *argv: (331, [[0], [make_save_pos(obs, get_enemy_pos_y_min_max(obs)[1])[0],
                                    make_save_pos(obs, get_enemy_pos_y_min_max(obs)[1])[1]]])
    if FUNCTIONS.Attack_screen.id in obs.observation.available_actions else (0, [])
)

# no op
tactic_no_op = Tactic(
    lambda *argv: True,
    lambda *argv: (0, []),
    lambda *argv: (0, [])
)
