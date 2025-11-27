from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import timeit
import numpy as np



# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
   
   
    H_robot = [0,0]
    robot=[]
    robot.append(env.get_robot(robot_id))
    robot.append(env.get_robot((robot_id + 1) % 2))
    packages = env.packages
    charge_stations = env.charge_stations
    pac_on_board = [pac for pac in packages if pac.on_board]
    A = [0,0]
    B = [0,0]
    C = [0,0]
    D = [0,0]
    
    for j in range(2):
   
        if robot[j].package is None:
            robot_to_package = [0] * len(pac_on_board)
            package_to_dest = [0] * len(pac_on_board)
            for i in range(len(pac_on_board)):
                robot_to_package[i] = manhattan_distance(robot[j].position, pac_on_board[i].position)
                package_to_dest[i] = manhattan_distance(pac_on_board[i].destination, pac_on_board[i].position)
            min_index_rob2pac = robot_to_package.index(min(robot_to_package))
            shortest_drive_to_package = robot_to_package[min_index_rob2pac]
            drive_pac2dest = package_to_dest[min_index_rob2pac]
            if robot[j].battery > (shortest_drive_to_package + drive_pac2dest):
                A[j] = robot[j].battery - shortest_drive_to_package
    
        if robot[j].package is not None:
            package_drive = manhattan_distance(robot[j].position, robot[j].package.destination)
            if package_drive < robot[j].battery:
                B[j] =  robot[j].battery - package_drive + manhattan_distance(robot[j].package.position, robot[j].package.position)
    
        if robot[j].credit > 0:
            dest_to_charge = [0] * len(charge_stations)
            for i in range(len(charge_stations)):
                dest_to_charge[i] = manhattan_distance(robot[j].position, charge_stations[i].position)
            # min_index_rob2pac = robot_to_package.index(min(robot_to_package))
            min_index_rob2charge = dest_to_charge.index(min(dest_to_charge))
            close_charge = dest_to_charge[min_index_rob2charge]
            if robot[j].package is not None:
                if robot[j].battery <= package_drive:
                    C[j] = robot[j].battery - close_charge  # +1
            if robot[j].package is None:
                if robot[j].battery < (shortest_drive_to_package + drive_pac2dest):
                    D[j] = robot[j].battery - close_charge
    
        H_robot[j] = A[j] + B[j] + C[j] + D[j]  + 5* robot[j].credit + 2* robot[j].battery

    return H_robot[0] - H_robot[1]


def calc_known_steps(env: WarehouseEnv, robot_id):
    robot = env.get_robot(robot_id)
    packages = env.packages
    pac_on_board = [pac for pac in packages if pac.on_board]


    if robot.package is not None and len(pac_on_board) == 1:
        return 3+manhattan_distance(robot.position, robot.package.destination)+manhattan_distance(robot.package.destination, pac_on_board[0].position)+manhattan_distance(pac_on_board[0].position, pac_on_board[0].destination)
    elif robot.package is not None and len(pac_on_board) == 0:
        return 1+manhattan_distance(robot.position, robot.package.destination)
    elif robot.package is None and len(pac_on_board) == 1:
        return 2+manhattan_distance(robot.position, pac_on_board[0].position)+manhattan_distance(pac_on_board[0].position, pac_on_board[0].destination)
    elif robot.package is None and len(pac_on_board) == 2:
        MD = [0]*len(pac_on_board)
        for i in range(len(pac_on_board)):
            MD[i] = manhattan_distance(robot.position, pac_on_board[i].position)
        closer_passenger = np.argmin(MD)
        return 4+manhattan_distance(robot.position, pac_on_board[closer_passenger].position)+manhattan_distance(pac_on_board[closer_passenger].position, pac_on_board[closer_passenger].destination)+manhattan_distance(robot.position, pac_on_board[1-closer_passenger].position)+manhattan_distance(pac_on_board[1-closer_passenger].position, pac_on_board[1-closer_passenger].destination)



class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1
    # def run_step(self, env: WarehouseEnv, agent_id, time_limit):
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start = timeit.default_timer()
        global initial_agent
        global max_depth
        global num_steps_left_ref
        num_steps_left_ref = 0
        max_depth=0
        initial_agent = agent_id
        depth = 1
        best_op = self.Play(env, agent_id, time_limit, depth)
        end = timeit.default_timer()
        time_used = end - start
        num_steps_left = calc_known_steps(env, agent_id)
        time_cond = 0.1 * time_limit**(depth+1)
        while time_used < time_cond and num_steps_left > depth +1:
            time_cond = 0.1 * time_limit ** (depth + 1)
            depth = depth + 1
            if max_depth<depth:
                max_depth=depth
                num_steps_left_ref = num_steps_left
            best_op = self.Play(env, agent_id, time_limit, depth)
            end = timeit.default_timer()
            time_used = end - start
        print(f'\n\n\nmax_depth: {max_depth}\{num_steps_left_ref}\n\n\n')
        return best_op

    def minimax(self ,env: WarehouseEnv, agent_id, time_limit, depth):
        robot = env.get_robot(agent_id)
        if depth == 0 or robot.battery == 0 or env.num_steps == 0:
            return smart_heuristic(env, initial_agent)
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        if agent_id == initial_agent:
            CurMax = -np.Inf
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
                v = self.minimax(child, 1 - agent_id, time_limit, depth - 1)
                CurMax = max(v, CurMax)
            return CurMax
        else:  # agent_id == 1
            CurMin = np.Inf
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
                v = self.minimax(child, 1 - agent_id, time_limit, depth - 1)
                CurMin = min(v, CurMin)

            return CurMin

    def Play(self, env: WarehouseEnv, agent_id, time_limit, depth):
        MaxV = -np.Inf
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
            v = self.minimax(child, 1 - agent_id, time_limit, depth - 1)
            if v >= MaxV:
                MaxV = v
                BestMove = op
        return BestMove




        # raise NotImplementedError()


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    # def run_step(self, env: WarehouseEnv, agent_id, time_limit):
    #     raise NotImplementedError()

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start = timeit.default_timer()
        global initial_alpha
        global max_depth
        global num_steps_left_ref
        num_steps_left_ref=0
        max_depth=0
        initial_alpha = agent_id
        depth = 1
        best_op = self.Play(env, agent_id, time_limit, depth)
        time_used = timeit.default_timer() - start
        num_steps_left = calc_known_steps(env, agent_id)
        time_cond = 0.1 * time_limit**(depth+1)
        while time_used < time_cond and num_steps_left > depth + 1:  # 300ms
            time_cond = 0.1 * time_limit ** (depth + 1)
            depth = depth + 1
            if max_depth<depth:
                max_depth=depth
                num_steps_left_ref=num_steps_left
            best_op = self.Play(env, agent_id, time_limit, depth)
            end = timeit.default_timer()
            time_used = end - start
        print(f'\n\n\nmax_depth: {max_depth}\{num_steps_left_ref}\n\n\n')
        return best_op

    def AlphaBeta(self, env: WarehouseEnv, agent_id, time_limit, depth, alpha, beta):
        robot = env.get_robot(agent_id)
        if depth == 0 or robot.battery == 0 or env.num_steps == 0:
            return smart_heuristic(env, initial_alpha)
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        if agent_id == initial_alpha:
            CurMax = -np.Inf
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
                v = self.AlphaBeta(child, 1-agent_id, time_limit, depth-1, alpha, beta)
                CurMax = max(v, CurMax)
                alpha = max(CurMax, alpha)
                if CurMax >= beta:
                    return np.Inf
            return CurMax
        else:
            CurMin = np.Inf
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
                v = self.AlphaBeta(child, 1-agent_id, time_limit, depth-1, alpha, beta)
                CurMin = min(v, CurMin)
                beta = min(CurMin, beta)
                if CurMin <= alpha:
                    return -np.Inf
            return CurMin

    def Play(self, env: WarehouseEnv, agent_id, time_limit, depth):
        alpha = -np.Inf
        Maxalpha = -np.Inf
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
            v = self.AlphaBeta(child, 1 - agent_id, time_limit, depth - 1, alpha, np.Inf)
            if v >= Maxalpha:
                alpha = v
                Maxalpha = v
                BestMove = op

        return BestMove


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)