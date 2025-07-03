# Shellbert

<img src="https://img.shields.io/badge/Status-Beta-#FDFD96" alt="Status" height="40">
<img src="https://img.shields.io/badge/Currently_Working_On-Personality_Engineering-8A2BE2" alt="Currently Working On" height="30">

Shellbert is an LLM _with a personality_ that will serve primarily two functions:

1. **Answering questions** related to moral/philosophy, economics, global health, animal welfare, and other topics covered in the course [Pathways to Progress](https://arizona.campusgroups.com/tea/about/).
2. **Automatically finding and saving impact-relevant jobs** and posting a list of curated opportunities in [Tucson Effective Altruism's](https://linktr.ee/tea_at_ua) discord server.

To better perform these two core tasks, the model will have the capacity to perform web searches, perform simple agentic tasks, and use tools.

Shellbert is currently based on the instruction-tuned versions of the [Gemma 3n models](https://huggingface.co/collections/google/gemma-3n-685065323f5984ef315c93f4). The model is hosted on a personal remote server with four NVIDIA 3060's (although, the model will typically use just a single GPU).

## Other Notes

- I am using [Cursor's agent](https://docs.cursor.com/chat/agent) to engineer this project; however, model selection, google and discord API setup, datasets for alignment and personality engineering, architectural and design decisions, and a fair amount of debugging has been or is being done by me.
- Currently, the model can be deployed on discord and has the capacity access previous conversations in the same channel, save user-specific context, and reply back-and-forth to messages; however, a lot of the alignment/personality scripts are quite immature and will undergo significant revisions.
- My core motivations behind this project are learning LLM personality engineering, safety fine-tuning, and running more evals. A secondary objective is improving my AI engineering and systems design skills. Lastly, I think the idea is fun; I came up with shellbert early in 2023, and it is fun to bring him to life!

## 🚀 Deployment guide

### 0. Pre-requisites

1. A **huggingface token** for downloading the Gemma 3n models. Accept the terms and conditions for the models [on huggingface](https://huggingface.co/collections/google/gemma-3n-685065323f5984ef315c93f4) before running any scripts.
2. **A discord "application" API token** for deploying the LLM-based bot on the server. You will need to login to discord and create an application in the developer page, and also add the bot to your chosen discord server.
3. **Google spreadsheet and drive API keys**, which can be generated using the Google Cloud platform.
4. **(optional) [SLURM](https://slurm.schedmd.com/documentation.html) for job scheduling**; this is not necessary if you want to interact with the model on discord; however, for full model deployment with job scraping features, the scripts assume that SLURM has been set up.

### 1. Environment Setup

First, create a .env file:

```text
# shellbert's configuration file

# model selection
MODEL_SIZE=e2b # https://huggingface.co/google/gemma-3n-E2B-it

# deployment settings
RUN_MODE=deploy
DEBUG_MODE=false

# huggingface authentication
HUGGINGFACE_TOKEN={insert your token here}

# discord integration
DISCORD_BOT_TOKEN={insert your token here}
# the server where the jobs should be posted
DISCORD_CHANNEL=#shellbert-test-1

# Optional Discord settings
# DISCORD_CHANNEL_ID=0
# DISCORD_MEMORY_TIMEOUT_HOURS=2
# DISCORD_MEMORY_MAX_MESSAGES=10

# slurm settings
SLURM_PARTITION=gpu
SLURM_CPUS_PER_GPU=8
SLURM_MEM_PER_GPU=11G
SLURM_TIME_LIMIT=12:00:00
SLURM_AUTO_SUBMIT=true
```

After creating the `.env` file, use download `uv` by running the following in your terminal (if you have a windows, [reference this website](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create a python 3.12.8 virtual environment by running:

```bash
uv venv --python 3.12.8
```

Finally, downloading all packages:

```bash
uv pip install -r requirements.txt
```

### 2. Model Selection

Choose your deployment strategy:

For development, I am using the smaller Gemma 3n model. You can choose this model by running:

```bash
echo "MODEL_SIZE=e2b" > .env
echo "RUN_MODE=development" >> .env
```

For deployment, I am using the larger Gemma 3n model. You can choose this model by running:

```bash
echo "MODEL_SIZE=e4b" > .env  
echo "RUN_MODE=production" >> .env
```

### 3. Run

Assuming all the pre-requisities and set up steps are followed, you could demo all the systems by running:

```bash
python -m shellbert demo
```

If the demo runs successfully, you can run the discord bot using:

```bash
python -m shellbert discord
```

If your machine has SLURM for job allocation, you can run the full model with the job scraping features as follows:

```bash
python -m shellbert
```