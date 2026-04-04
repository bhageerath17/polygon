.PHONY: setup fetch analyze reversal run serve clean

setup:
	uv sync --all-extras
	@echo "✓ Environment ready. Activate with: source .venv/bin/activate"

fetch:
	PYTHONPATH=. uv run python scripts/fetch.py

analyze:
	PYTHONPATH=. uv run python scripts/analyze.py

reversal:
	PYTHONPATH=. uv run python scripts/reversal_analysis.py

run:
	PYTHONPATH=. uv run python scripts/backtest.py

serve:
	cp data/spx_1min.csv data/spx_options_snapshot.csv data/patches_analysis.json data/reversal_analysis.json dashboard/
	@echo "Open http://localhost:8000/dashboard/"
	uv run python -m http.server 8000

clean:
	rm -rf .venv __pycache__ **/__pycache__ **/*.pyc
