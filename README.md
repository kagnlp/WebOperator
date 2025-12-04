<div align="center">
<img src="assets/wo_logo_2.png" alt="WEB-OPERATOR Logo"/>
<h1>WebOperator: Action-Aware Tree Search for Autonomous Agents in Web Environment</h1>
</div>

![WebOperator](assets/wo_banner.svg)

## 📖 Abstract

LLM-based agents often operate in a greedy, step-by-step manner, selecting actions solely based on the current observation without considering long-term consequences or alternative paths. 
This lack of foresight is particularly problematic in web environments, which are only partially observable—limited to browser-visible content such as the current page’s DOM and UI elements—where a single misstep often requires complex and brittle navigation to undo. Without an explicit backtracking mechanism, agents struggle to correct errors or systematically explore alternative paths. Tree-search methods provide a principled framework for such structured exploration, but existing approaches lack mechanisms for safe backtracking, making them prone to unintended side effects. They also assume that all actions are reversible, ignoring the presence of irreversible actions—limitations that reduce their effectiveness in realistic web tasks. To address these challenges, we introduce **WebOperator**, a tree-search framework that enables reliable backtracking and strategic exploration. Our method incorporates a best-first search strategy that ranks actions by both reward estimates and safety considerations, along with a robust backtracking mechanism that verifies the feasibility of previously visited paths before replaying them, preventing unintended side effects. To further guide exploration, WebOperator generates action candidates from multiple, varied reasoning contexts to ensure diverse and robust exploration, and subsequently curates a high-quality action set by filtering out invalid actions pre-execution and merging semantically equivalent ones. Experimental results on WebArena and WebVoyager demonstrate the effectiveness of WebOperator. Notably, on WebArena, WebOperator achieves state-of-the-art performance with gpt-4o, achieving a **54.56%** success rate, underscoring the critical advantage of integrating strategic foresight with safe execution.

## 📂 Project Structure

```bash
.
├── weboperator/                 # Source code for the web agent
├── webshepherd/                 # Source code for the Process Reward Model
├── browsergym/                  # Source code for the web environment simulator
├── gobrowse/                    # Source code for the experience retrieval module
└── README.md
```

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/kagnlp/WebOperator.git
cd WebOperator
```

### 2️⃣ Create environment
```bash
conda create -n weboperator_env python=3.12
conda activate weboperator_env
# or using pip and virtualenv
python -m venv weboperator_env
source weboperator_env/bin/activate  # On Windows use `weboperator_env\Scripts\activate`
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
playwright install chromium --with-deps # Need admin rights
```

### 4️⃣ Set up environment variables
Create a `.env` file by copying the example configuration:

```bash
cp .env.example .env
```

Then open the `.env` file and update any necessary values (such as API keys, website urls) according to your environment.

## 🚀 Usage

### Skeleton Code
Boilerplate code ([demo.py](demo.py)) to run WebOperator on an interactive, open-ended task:

```python
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
```

<!-- ## Evaluation -->

<!-- ### WebArena 

```bash
python run.py --config weboperator/configs/wa-gpt-oss-20b.yml
``` -->

<!-- # OR using docker
docker compose run --user $(id -u) weboperator --config weboperator/configs/wa-gpt-oss-20b.yml -->
<!-- 
- conda env create -f dev/environment.yaml
- pip install --upgrade playwright==1.54.0 && playwright install chromium --with-deps
- pip install --upgrade playwright==1.32.1 && playwright install chromium

pip install -e browsergym/core
pip install -e browsergym/experiments
pip install -e browsergym/webvoyager
pip install -e browsergym/webarena
pip install dotenv
pip install rank_bm25
pip install sentence_transformers
pip install faiss-cpu
pip install tabulate
pip install openai==1.97.1
pip install beartype
pip install nltk

playwright install chromium

conda install -c conda-forge mesa xorg-libxshmfence nss libxkbcommon
conda install -c conda-forge mesa-libgl-cos7-x86_64 mesa-dri-drivers-cos7-x86_64 -->

<!-- Create .env -->
<!-- numpy==1.26.4 -->
<!-- - pip install -e browsergym/miniwob # local package
- pip install -e browsergym/webarena # local package
- pip install -e browsergym/webvoyager # local package
- pip install -e browsergym/visualwebarena # local package
- pip install -e browsergym/assistantbench # local package
   -->
<!-- 
gitlab -> something went wrong on our end.
reddit -> 504 Gateway Time-out -->