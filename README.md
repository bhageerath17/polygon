# SPX Options Dashboard

A data pipeline and interactive dashboard for SPX options and price data, powered by [Polygon.io](https://polygon.io).

Live dashboard → **[bhageerath17.github.io/polygon](https://bhageerath17.github.io/polygon)**

---

## Architecture

```mermaid
graph TD
    A[Polygon.io API] -->|1-min OHLC| B[fetch_spx_data.py]
    A -->|Options Snapshot| B
    B -->|spx_1min_jan2026.csv| C[Data Layer]
    B -->|spx_options_snapshot.csv| C
    C --> D[dashboard/index.html]
    D --> E[GitHub Pages]

    style A fill:#238636,color:#fff
    style E fill:#1f6feb,color:#fff
```

---

## Data Flow

```mermaid
sequenceDiagram
    participant Dev as Developer
    participant Script as fetch_spx_data.py
    participant API as Polygon.io
    participant Dash as Dashboard

    Dev->>Script: make fetch
    Script->>API: GET /v2/aggs/ticker/I:SPX (1-min bars)
    API-->>Script: 7,867 bars (Jan 2026)
    Script->>API: GET /v3/snapshot/options/I:SPX
    API-->>Script: Options chain (greeks, OI, IV)
    Script-->>Dev: CSVs written locally
    Dev->>Dash: make serve
    Dash-->>Dev: http://localhost:8000/dashboard/
```

---

## Project Structure

```mermaid
graph LR
    root[polygon/]
    root --> fetch[fetch_spx_data.py]
    root --> backtest[polygon_backtest.py]
    root --> dash[dashboard/]
    dash --> html[index.html]
    root --> pyproject[pyproject.toml]
    root --> mk[Makefile]
    root --> env[.env 🔒]

    style env fill:#3b0f0f,color:#f85149
```

---

## Backtesting Pipeline _(coming soon)_

```mermaid
flowchart LR
    A[SPX 1-min Bars] --> B{Signal}
    B -->|Entry| C[Open Position]
    C --> D{Exit Condition}
    D -->|Stop / Target / Expiry| E[Close Position]
    E --> F[P&L Log]
    F --> G[Dashboard]

    style B fill:#161b22,color:#e6edf3
    style G fill:#1f6feb,color:#fff
```

---

## Quick Start

### Prerequisites
- [uv](https://docs.astral.sh/uv/) — fast Python package manager
- A [Polygon.io](https://polygon.io) API key

### Setup

```bash
# 1. Clone
git clone https://github.com/bhageerath17/polygon.git
cd polygon

# 2. Copy and fill in your API key
cp .env.example .env

# 3. Install dependencies
make setup

# 4. Fetch SPX data
make fetch

# 5. Open the dashboard locally
make serve
# → http://localhost:8000/dashboard/
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `POLYGON_API_KEY` | Your Polygon.io API key |

Create a `.env` file (never commit it):
```
POLYGON_API_KEY=your_key_here
```

---

## Makefile Commands

| Command | Description |
|---|---|
| `make setup` | Create `.venv` and install all dependencies |
| `make fetch` | Pull latest SPX price + options data |
| `make serve` | Serve dashboard at `localhost:8000` |
| `make run` | Run the backtest script |
| `make clean` | Remove `.venv` and caches |

---

## Dashboard

Shows live SPX insights pulled from Polygon.io:

- **SPX Jan 2026 summary** — open, close, high, low, % move
- **1-min bar count** for the month
- **Average implied volatility** across the options chain
- **Total open interest** split by calls vs puts
- **Top 20 options by open interest** — expiry, strike, greeks, bid/ask
