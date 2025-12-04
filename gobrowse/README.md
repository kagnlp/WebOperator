## Download Dataset

```bash
git clone https://huggingface.co/datasets/mahirlabibdihan/weboperator-go-browse.git websites
python patch.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Run RAG Server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```