.PHONY: help install test lint docker-build docker-up docker-down docker-logs docker-shell docker-test

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

docker-build:
	docker compose build

docker-up:
	docker compose up --build

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f jupyter

docker-shell:
	docker compose run --rm jupyter bash
