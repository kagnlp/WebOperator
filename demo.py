import gymnasium as gym
import browsergym.core  # register the openended task as a gym environment
from weboperator.tree_search_agent import TreeSearchAgent
from weboperator.action_generator import ActionGenerator
from weboperator.models.openrouter import OpenRouterModel

# start an openended environment
env = gym.make(
    "browsergym/openended",
    task_kwargs={"start_url": "https://map.google.com/"},  # starting URL
    wait_for_user_message=True,  # wait for a user message after each agent message sent to the chat
    headless=False
)

# Create an agent
action_generator = ActionGenerator(
    model=OpenRouterModel("openai/gpt-oss-20b:free")  # Set OPENROUTER_API_KEYS in .env file
)
agent = TreeSearchAgent(
        chat_mode=True,
        action_generator= action_generator,
    )

# run the environment <> agent loop until termination
obs, info = env.reset()
while True:
    preprocessed_obs = agent.obs_preprocessor(obs) # Preprocess observation
    action = agent.get_action(preprocessed_obs, env) # Decide action
    obs, reward, terminated, truncated, info = env.step(action) # Act and Observe
    if terminated or truncated:
        break
# release the environment
env.close()