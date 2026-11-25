.PHONY: up down logs test clean

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f

test:
	docker compose run --rm backend python -m pytest tests/ -v

clean:
	docker compose down -v --remove-orphans
