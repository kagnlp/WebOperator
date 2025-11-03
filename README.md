<div align="center">
<img src="assets/wo_logo_2.png" alt="WEB-OPERATOR Logo"/>
<h1>WebOperator: Action-Aware Tree Search for Autonomous Agents in Web Environment</h1>
</div>

![WebOperator](assets/wo_banner.svg)

## 📖 Abstract

Conventional WebAgents often operate in a greedy, step-by-step manner, selecting actions based solely on the current observation without considering long-term consequences or alternative paths. This lack of foresight is particularly problematic in web environments, which are only partially observable—limited to browser-visible content such as the current page’s DOM and UI elements—where a single misstep can drive the agent into a dead-end state, from where the goal is unreachable. Without a mechanism for backtracking, the agent has no way to correct such errors or explore alternative paths once a mistake has been made. Tree search methods provide a principled framework for such structured exploration, but existing approaches assume all actions are safe and reversible, ignore irreversible actions, and suffer from inefficient backtracking—limitations that reduce their effectiveness in realistic web tasks. To address these challenges, we introduce **WebOperator**, a tree-search framework for reliable and efficient exploration of web environments. Our method incorporates an action-aware Best-First Search strategy that ranks actions by both reward estimates and safety considerations, along with a robust backtracking mechanism that validates path feasibility through isolated simulation before committing changes to the main environment—preventing errors and minimizing redundant steps. In addition, WebOperator integrates strategies for guided exploration, experience retrieval that leverages past interactions, and semantic merging to eliminate redundant actions, enabling systematic completion of complex web tasks. Experiments on WebArena and WebVoyager show that even with weaker open-source backbone LLMs, our method outperforms current state-of-the-art approaches with proprietary strong models, demonstrating its effectiveness for autonomous agents in web environments.

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

## 🚀 Usage

### Skeleton Code
Boilerplate code to run WebOperator on an interactive, open-ended task:

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
    model=OpenRouterModel("openai/gpt-oss-20b:free")
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

## Evaluation

### WebArena 

```bash
python run.py --config weboperator/configs/wa-gpt-oss-20b.yml
# OR using docker
docker compose run --user $(id -u) weboperator --config weboperator/configs/wa-gpt-oss-20b.yml
```

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