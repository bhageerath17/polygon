.PHONY: setup fetch fetch-intraday fetch-vix1d analyze reversal model model2 model3 warlord \
       fetch-all analyze-all models-all build-all run serve clean

# ── Individual targets ────────────────────────────────────────────────────
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

model2:
	PYTHONPATH=. uv run python scripts/build_model2.py

model3:
	PYTHONPATH=. uv run python scripts/build_model3.py

warlord:
	PYTHONPATH=. uv run python scripts/build_warlord.py

# ── Parallel phase targets (mirrors CI) ───────────────────────────────────
fetch-all:                          ## Phase 1: fetch + fetch-intraday in parallel
	$(MAKE) -j2 fetch fetch-intraday

analyze-all: fetch-all              ## Phase 2: vix1d + analyze + reversal in parallel
	$(MAKE) -j3 fetch-vix1d analyze reversal

models-all: analyze-all             ## Phase 3: all 4 models in parallel
	$(MAKE) -j4 model model2 model3 warlord

build-all: models-all               ## Full pipeline: fetch → analyze → models (parallel within each phase)
	@echo "✓ Full build complete"

# ── Dev helpers ───────────────────────────────────────────────────────────
run:
	PYTHONPATH=. uv run python scripts/backtest.py

serve: build-all
	cp data/spx_1min.csv data/spx_options_snapshot.csv data/patches_analysis.json data/reversal_analysis.json data/model_results.json dashboard/
	-cp data/model2_results.json data/model3_results.json data/warlord_results.json dashboard/ 2>/dev/null
	@echo "Open http://localhost:8000/dashboard/"
	uv run python -m http.server 8000

clean:
	rm -rf .venv __pycache__ **/__pycache__ **/*.pyc
