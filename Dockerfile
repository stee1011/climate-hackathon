# ─────────────────────────────────────────────
# DOCKERFILE — Scenario Forecasting Module
# ─────────────────────────────────────────────
#
# What this file does in plain English:
#   Docker uses this as a recipe to build a
#   self-contained box (container) that has
#   everything needed to run your module —
#   Python, all libraries, your code, and the data.
#
#   Anyone on the team can run this box without
#   installing anything on their own computer.
#   It works identically on every machine.
#
# How it fits with the team's setup:
#   Your teammate's docker-compose.yml already
#   defines services for frontend, api, and streamlit.
#   This Dockerfile builds YOUR service, which gets
#   added as a new entry in that compose file.
# ─────────────────────────────────────────────

# START FROM a clean Python 3.11 image
# "slim" means it's a smaller version — no extras we don't need
FROM python:3.11-slim

# SET the working directory inside the container
# All commands from here run inside /app
WORKDIR /app

# INSTALL system dependencies
# These are needed by some Python scientific libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# COPY requirements file first (before the rest of the code)
# Why? Docker caches each step. If requirements don't change,
# it won't reinstall everything just because you edited your code.
COPY requirements.txt .

# INSTALL Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# COPY your module files into the container
COPY scenario_forecasting.py .
COPY scenario_api.py .

# NOTE: data.csv is mounted at runtime via docker-compose
# so the team shares one copy of the data across all services
# (see docker-compose.yml volumes section)

# EXPOSE the ports this container uses
# 8501 = Streamlit UI
# 8502 = FastAPI scenario API
EXPOSE 8501
EXPOSE 8502

# CREATE a startup script that runs BOTH services simultaneously
# Streamlit runs in the background (&), FastAPI runs in the foreground
RUN echo '#!/bin/bash\n\
streamlit run scenario_forecasting.py \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --browser.gatherUsageStats=false &\n\
uvicorn scenario_api:app --host 0.0.0.0 --port 8502\n\
' > /app/start.sh && chmod +x /app/start.sh

# DEFAULT COMMAND — run the startup script
CMD ["/bin/bash", "/app/start.sh"]
