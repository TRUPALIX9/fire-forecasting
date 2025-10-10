# Fire Forecasting Makefile - Frontend Only

.PHONY: help run-frontend clean

help:
	@echo "Available targets:"
	@echo "  run-frontend     - Start the Next.js frontend server"
	@echo "  clean            - Clean up generated files"

run-frontend:
	cd frontend && npm run dev

clean:
	find . -name "*.log" -delete
