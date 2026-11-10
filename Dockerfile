FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# приложение работает от непривилегированного пользователя
RUN groupadd --system app && useradd --system --gid app --home /app --shell /usr/sbin/nologin app
WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install ".[dashboard,orchestration]"

COPY config.yaml ./
COPY sql ./sql
COPY dashboard ./dashboard
RUN mkdir -p /app/artifacts /app/data/inbox/rosstat && chown -R app:app /app/artifacts /app/data

USER app
CMD ["tourflow", "--help"]
