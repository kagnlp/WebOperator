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
_Refer to the [Running with Docker](#running-with-docker) section if you don't have admin rights to install Playwright dependencies._
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

### Run the Demo

```bash
python demo.py
```

**or**

```bash
python run.py --config weboperator/configs/default.yml
```

#### 🐳 Running with Docker
_Useful if you don't have admin rights to install Playwright dependencies. No need to create a virtual environment or install dependencies._

```bash
docker compose run --user $(id -u) weboperator --config weboperator/configs/default.yml
```

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

## 📊 Benchmark Configurations

#### WebArena

```bash
python run.py --config weboperator/configs/wa-gpt-4o.yml
```

#### WebVoyager

```bash
python run.py --config weboperator/configs/wv-gpt-4o.yml
```

## ⚙️ Agent Configuration Explanation

#### Environment

```yaml
env:
  task_type: "openended" # ["webarena", "webvoyager", "openended"]
  max_steps: 100 # Maximum steps per episode (For BrowserGym)
  headless: false # false: show browser UI; true: hide browser UI
```

#### Experiment

```yaml
experiment:
  results_dir: "./results/openended/gpt-oss-20b" # Directory to save results. Give relative path.
```

#### Agent

```yaml
agent:
  allow_unauthorized_page: true # Whether allow visit to pages outside the benchmark domain
```

#### Models

```yaml
models: # List of models used in the agent
  action_model: # Unique identifier of the model
    type: "OpenRouterModel" # Options: ["OpenAIModel", "AzureOpenAIModel", "OpenRouterModel", "OpenHFModel"]
    model_name: "openai/gpt-oss-20b:free"
  reward_model:
    type: "AzureOpenAIModel"
    model_name: "gpt-4o"
    temperature: 1.0
```

#### Agent Components

```yaml
components:
  action_validator: # Optional: Action validator configuration
    allow_invalid_action: false # Whether to allow semantically invalid actions (Default: false)
    allow_invalid_page: false # Whether to allow navigation to invalid pages (Default: false)

  observation_processor: # Observation processor configuration
    optimized: true # true: use full or visible-only observation based on the observation size. false: always use visible-only observation
    truncate_error_message: true # Truncate long error messages

  action_processor: # Action processor configuration
    merge_strategy: "sum" # ["sum", "max", "none"]: strategy to merge semantically similar actions. "none": do not merge.
  
  recovery_assistant: # Optional: Recovery assistant configuration
    recover_from_invalid_page: true # true: forcefully go_back or tab_close when on invalid page
    recover_from_captcha: true # Whether to allow human intervention for captcha recovery

  backtrack_manager: # Optional: Enables backtracking mechanism
    destruction_aware: true # Whether to re-root the tree after executing destructive actions
    simulation_verified: true # Whether to do snapshot-validation or not

  action_selector: # Action selection strategy configuration
    selection_strategy: "action-aware" # options: ["highest-reward", "action-aware"]
    search_budget: 4 # Frontier budget
    n_candidates: 2 # Number of solution candidates to consider
    max_depth: 20 # Maximum search depth
    max_steps: 20 # Maximum steps (excluding backtracking steps)

  rephraser: # Optional: Enables instruction rephraser
    model: "action_model" # Model used for rephrasing instructions

  retriever: # Optional: Enables examples retriever
    type: "faiss" # ["faiss", "bm25"]
    model: "all-MiniLM-L6-v2" # Sentence transformer model (for faiss retriever)
    top_k: 5 # Number of examples to retrieve

  judge: # Reward and checklist model configuration. Note: Applicable only for multiple action candidates
    prompt_type: "web_operator"  # Options: likert_scale, web_shepherd, web_operator
    checklist_model: "reward_model" # Model used for checklist generation
    reward_model: "reward_model" # Model used for reward estimation

  action_generator:
    max_retry: 5 # Maximum retries for generating syntactically and semantically valid actions
    full_action_space: # List of all possible actions
      - "click"
      - "fill"
      - "select_option"
      - "goto"
      - "go_back"
      - "go_forward"
      - "scroll"
      - "new_tab"
      - "tab_focus"
      - "tab_close"
      - "stop"
    action_space_type: "adaptive" # options: ["fixed", "adaptive"]
    candidates: # List of action generator candidates
      - name: "simple_action_generator" # Unique name for the candidate
        model: "action_model" # Model to use 
        history_length: 5 # Number of previous steps to include in the context
        rephraser: false # Whether to include rephrased task instruction
        retriever: false # Whether to include retrieved examples
      - name: "action_generator_w_retriever"
        model: "action_model"
        history_length: 3
        rephraser: false
        retriever: true
      - name: "action_generator_w_rephraser"
        model: "action_model" 
        history_length: 4
        rephraser: true
        retriever: false
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