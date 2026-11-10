.PHONY: up down pipeline demo logs status password test lint audit

up:            ## поднять БД, Prefect, планировщик и дашборд
	docker compose up -d --build

down:
	docker compose down

pipeline:      ## разовый полный прогон: загрузка → проверки → признаки → модели
	docker compose run --rm pipeline

demo:          ## тот же прогон на синтетических данных (без сети)
	docker compose run --rm -e TOURFLOW_SOURCE_MODE=synthetic pipeline

status:
	docker compose run --rm pipeline tourflow status

logs:
	docker compose logs -f --tail=100 scheduler dashboard

password:      ## хеш пароля дашборда для .env
	docker compose run --rm --no-deps pipeline tourflow hash-password

test:
	pytest

lint:
	ruff check src tests dashboard
	bandit -q -r src -x tests

audit:         ## известные уязвимости в зависимостях
	pip-audit
