.PHONY: setup fetch fetch-intraday fetch-vix1d analyze reversal model run serve clean

setup:
	uv sync --all-extras
	@echo "✓ Environment ready. Activate with: source .venv/bin/activate"

fetch:
	PYTHONPATH=. uv run python scripts/fetch.py

fetch-intraday:
	PYTHONPATH=. uv run python scripts/fetch_intraday.py

fetch-vix1d:
	PYTHONPATH=. uv run python scripts/fetch_vix1d.py

analyze:
	PYTHONPATH=. uv run python scripts/analyze.py

reversal:
	PYTHONPATH=. uv run python scripts/reversal_analysis.py

model:
	PYTHONPATH=. uv run python scripts/build_model.py

run:
	PYTHONPATH=. uv run python scripts/backtest.py

serve:
	cp data/spx_1min.csv data/spx_options_snapshot.csv data/patches_analysis.json data/reversal_analysis.json data/model_results.json dashboard/
	@echo "Open http://localhost:8000/dashboard/"
	uv run python -m http.server 8000

clean:
	rm -rf .venv __pycache__ **/__pycache__ **/*.pyc
