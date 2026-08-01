import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Try to import stable_baselines3 and hisss
try:
    from stable_baselines3 import PPO
except ImportError:
    print("Please install stable_baselines3: pip install stable-baselines3")
    sys.exit(1)

try:
    import hisss
    from hisss.game.state import BattleSnakeState
except ImportError:
    print("Please install hisss: pip install hisss")
    sys.exit(1)

from battlesnake_types import GameState, MoveAction, Direction, BaseAgent
from ppo_training.config import ExperimentConfig
from ppo_training.observation import ExplicitMemory, ObservationBuilder
import torch

# Limit PyTorch to 1 thread to avoid memory fragmentation/SIGABRT (status 134) on constrained environments like Render
torch.set_num_threads(1)

logger = logging.getLogger(__name__)


class PPOAgent(BaseAgent):
    def __init__(self, model_path: str, config_path: str):
        super().__init__()
        self.model_path = model_path
        self.config_path = config_path
        
        # Load configuration
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config not found at {self.config_path}")
        self.config = ExperimentConfig.load(self.config_path)
        
        # Initialize observation builder
        self.obs_builder = ObservationBuilder(self.config)
        
        # Load PPO Model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        print(f"Loading PPO model from {self.model_path}...")
        # Load on CPU for inference to avoid CUDA overhead on web servers
        self.model = PPO.load(self.model_path, device="cpu")
        print("Model loaded successfully.")
        
        # Memories for Fog of War (game_id -> ExplicitMemory)
        self.memories = {}

    def get_name(self):
        return "PPO Blackout Agent"

    def get_color(self):
        return '#800080'  # Purple

    def get_author(self):
        return "RL_Trainer"

    def start(self, game_state: GameState):
        """start is called when the battlesnake begins a game"""
        self.memories[game_state.game.id] = ExplicitMemory()
        
    def end(self, game_state: GameState):
        """end is called when the battlesnake finishes a game"""
        if game_state.game.id in self.memories:
            del self.memories[game_state.game.id]

    def move(self, game_state: GameState) -> MoveAction:
        """move is called on every turn and returns your next move"""
        game_id = game_state.game.id
        if game_id not in self.memories:
            self.memories[game_id] = ExplicitMemory()
            
        memory = self.memories[game_id]
        
        game_config = hisss.restricted_standard_config()
        game_config.h = game_state.board.height
        game_config.w = game_state.board.width
        game_config.all_actions_legal = True
        
        # Override config settings if needed
        if hasattr(game_state.game.ruleset, 'settings') and hasattr(game_state.game.ruleset.settings, 'viewRadius'):
            game_config.view_radius = game_state.game.ruleset.settings.viewRadius or self.config.view_radius
        
        env = hisss.BattleSnakeGame(game_config)
        
        snakes_alive = []
        snake_pos = {}
        snake_health = []
        snake_len = []
        
        # Order snakes: YOU are always index 0
        snakes_list = [game_state.you]
        for s in game_state.board.snakes:
            if s.id != game_state.you.id:
                snakes_list.append(s)
                
        num_players = game_config.num_players
        alive_count = 0
        
        for i in range(num_players):
            if i < len(snakes_list):
                s = snakes_list[i]
                is_alive = True
                
                # Filter out None values and out-of-bounds coordinates
                body = [
                    (p.x, p.y) for p in s.body 
                    if p is not None and 0 <= p.x < game_config.w and 0 <= p.y < game_config.h
                ]
                
                # If snake has no valid body parts, it's effectively dead to hisss
                if not body:
                    is_alive = False
                    body = []
                else:
                    alive_count += 1
                    # Pad body to match s.length to avoid C++ out-of-bounds read in hisss
                    # In Fog of War, Battlesnake might send only the visible parts of the body,
                    # but s.length reflects the true length.
                    while len(body) < s.length:
                        body.append(body[-1])
                    
                snakes_alive.append(is_alive)
                snake_pos[i] = body
                snake_health.append(s.health if s.health is not None else 0)
                snake_len.append(s.length)
            else:
                snakes_alive.append(False)
                snake_pos[i] = []
                snake_health.append(0)
                snake_len.append(0)
                
        # If there are fewer than 2 snakes alive, hisss will throw an error on get_obs()
        if alive_count < 2:
            print(f"[{game_id}] Turn {game_state.turn}: Only 1 snake alive. Falling back to UP.")
            return MoveAction(move=Direction.UP)

        food_pos = [
            [f.x, f.y] for f in game_state.board.food 
            if 0 <= f.x < game_config.w and 0 <= f.y < game_config.h
        ]
        
        state = BattleSnakeState(
            turn=game_state.turn,
            snakes_alive=snakes_alive,
            snake_pos=snake_pos,
            food_pos=food_pos,
            snake_health=snake_health,
            snake_len=snake_len
        )
        
        env.set_state(state)
        
        # Track newly announced food (food visible this turn)
        # For a simple web server deployment without history, we can just use all current food.
        # It's an approximation, but ObservationBuilder will update memory correctly.
        current_food_set = {(f.x, f.y) for f in game_state.board.food}
        
        try:
            obs, inverse_action_map = self.obs_builder.observe(
                env=env,
                player=0, # We placed 'you' at index 0
                memory=memory,
                announced_food=current_food_set,
                symmetry=0
            )
        except ValueError as e:
            if "terminal Hisss game" in str(e):
                print("CRASH DETECTED! DUMPING HISSS STATE:")
                print(f"snakes_alive: {snakes_alive}")
                print(f"snake_pos: {snake_pos}")
                print(f"snake_health: {snake_health}")
                print(f"snake_len: {snake_len}")
                print(f"food_pos: {food_pos}")
                print(f"w: {game_config.w}, h: {game_config.h}, num_players: {game_config.num_players}")
                import sys
                sys.stdout.flush()
                # Fallback to UP if hisss fails to avoid crashing the whole turn (though C++ might still crash on GC)
                return MoveAction(move=Direction.UP)
            raise
        
        # Predict move
        # PPO predict expects a batched observation, so we add a batch dimension
        action_idx, _ = self.model.predict(np.expand_dims(obs, axis=0), deterministic=True)
        
        # Extract scalar
        action_idx = int(action_idx.item())
        
        # Map integer to Direction
        # In hisss, actions are: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        action_map = {
            int(hisss.UP): Direction.UP,
            int(hisss.RIGHT): Direction.RIGHT,
            int(hisss.DOWN): Direction.DOWN,
            int(hisss.LEFT): Direction.LEFT
        }
        
        original_action = inverse_action_map[action_idx]
        chosen_direction = action_map.get(original_action, Direction.UP)
        
        print(f"[{game_id}] Turn {game_state.turn}: PPO chose {chosen_direction.value}")
        return MoveAction(move=chosen_direction)


if __name__ == "__main__":
    import sys
    from battlesnake_server import start_server

    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <port>")
        sys.exit(1)

    # Initialize agent
    model_path = os.path.join(os.path.dirname(__file__), "ppo_model.zip")
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    
    agent = PPOAgent(model_path=model_path, config_path=config_path)
    
    port = int(sys.argv[1])
    start_server(agent=agent, port=port)
