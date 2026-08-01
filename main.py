import os
from battlesnake_server import start_server
from ppo_agent import PPOAgent

if __name__ == "__main__":
    # Initialize the PPO agent
    model_path = os.path.join(os.path.dirname(__file__), "ppo_model.zip")
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    
    agent = PPOAgent(model_path=model_path, config_path=config_path)
    
    # Render provides the port in the PORT environment variable
    port = int(os.environ.get("PORT", "8080"))
    start_server(agent=agent, port=port)