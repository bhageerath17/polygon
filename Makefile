.PHONY: setup run fetch serve clean

setup:
	uv sync --all-extras
	@echo "✓ Environment ready. Activate with: source .venv/bin/activate"

fetch:
	uv run python fetch_spx_data.py

run:
	uv run python polygon_backtest.py

serve:
	@echo "Open http://localhost:8000/dashboard/"
	uv run python -m http.server 8000

clean:
	rm -rf .venv __pycache__ *.pyc
