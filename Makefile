# Fire Forecasting Makefile - Frontend prototype

.PHONY: help install run-frontend build sample-data clean

help:
	@echo "Available targets:"
	@echo "  install          - Install frontend dependencies (npm ci)"
	@echo "  run-frontend     - Start the Next.js dev server on http://localhost:3000"
	@echo "  build            - Type check and build the frontend for production"
	@echo "  sample-data      - Regenerate frontend/lib/sample-forecast.json from data/trihourly_weather.csv"
	@echo "  clean            - Remove build output and log files"

install:
	cd frontend && npm ci

run-frontend:
	cd frontend && npm run dev

build:
	cd frontend && npm run typecheck && npm run build

sample-data:
	python3 scripts/build_sample_data.py

clean:
	rm -rf frontend/.next
	find . -name "*.log" -not -path "./frontend/node_modules/*" -delete
