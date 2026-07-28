import os
import time
from random_agent import RandomAgent
from hungry_agent import HungryAgent  # <-- 1. Import your actual agent!
from battlesnake_types import BaseAgent, GameState, Direction
import hisss

DIRECTION_TO_HISSS = {
    Direction.UP: hisss.UP,
    Direction.DOWN: hisss.DOWN,
    Direction.LEFT: hisss.LEFT,
    Direction.RIGHT: hisss.RIGHT
}

def run_game(
    agents: list[BaseAgent],
):
    game_cfg = hisss.restricted_standard_config()
    game_cfg.all_actions_legal = True  # by default, hisss disallows actions which immediately kill a player to reduce action space
    env = hisss.BattleSnakeGame(game_cfg)
    
    for idx, cur_agent in enumerate(agents):
        cur_str = hisss.to_battlesnake_json(env, idx)
        cur_state = GameState.model_validate_json(cur_str)
        cur_agent.start(cur_state)
    
    os.system('cls' if os.name == 'nt' else 'clear')
    env.render()
    
    while not env.is_terminal():
        moves: list[int] = []
        for idx, cur_agent in enumerate(agents):
            # skip dead snakes
            if not env.is_player_at_turn(idx):
                continue
            # export env state to json and call agent
            cur_str = hisss.to_battlesnake_json(env, idx, include_eliminated=True)
            cur_state = GameState.model_validate_json(cur_str)
            move_result = cur_agent.move(cur_state)
            moves.append(DIRECTION_TO_HISSS[move_result.move])
        
        env.step(actions=tuple(moves))
        
        # --- 2. ANIMATION LOGIC ADDED HERE ---
        os.system('cls' if os.name == 'nt' else 'clear') # Clears the terminal
        env.render()                                     # Prints the new board
        time.sleep(0.15)                                 # Pauses for 0.15 seconds
        # -------------------------------------
            
    for idx, cur_agent in enumerate(agents):
        cur_str = hisss.to_battlesnake_json(env, idx)
        cur_state = GameState.model_validate_json(cur_str)
        cur_agent.end(cur_state)
    
    elim_events = env.get_state().elimination_events
    if elim_events is not None:
        for idx, event in elim_events.items():
            print(f"Snake {idx}: {event}")
    print(f"\nSnakes alive at the end: {env.players_alive()}")
    
if __name__ == '__main__':
    
    agents: list[BaseAgent] = [
        HungryAgent(),   # <-- 3. Gluttony enters the arena! (Snake 0)
        RandomAgent(),   # Snake 1
        RandomAgent(),   # Snake 2
        RandomAgent(),   # Snake 3
    ]
    run_game(agents)
    

