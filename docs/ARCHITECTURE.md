# Architecture

AkashNetra AI is an **add-on** that runs after an NWP forecast. It does not forecast
weather; it estimates the probability that the rainfall forecast for a 1°×1° box and lead
day (1-10) will be badly wrong ("bust"), and explains why.

```mermaid
flowchart LR
    subgraph Sources
        G[GEFS v12 reforecast] --> U
        N[NCUM/NEPS adapter] --> U
        S[Synthetic source - DEMO] --> U
        I[IMD gridded rain] --> L
        E[ERA5 - regime tags] --> F
    end
    U[Unified tidy table\n init_date x lead_day x box] --> L[Bust labels\n train-only P90]
    L --> F[Features + analog search\n train years only]
    F --> M[LightGBM + XGBoost\n isotonic calibration]
    M --> X[SHAP + physics rules + reasons]
    M --> A[FastAPI GeoJSON]
    X --> A
    A --> D[Streamlit dashboard]
```

## Design decisions (living list)

| Decision | Why |
|---|---|
| Thresholds computed on train years only, stored as an artifact | Prevents leakage into val/test |
| Bust threshold per (box, lead day) within the season (`bust.per_lead_day: true`) | Spec says "per box per season"; errors grow with lead, so pooling leads would make long leads bust far more than 10%. Configurable; revisit if you prefer pooling. |
| Season assigned by init date | A forecast issued 30 Sep stays in JJAS even if it verifies in October |
| No automatic fallback from real to synthetic data | Honesty rule; `data_mode` is explicit and surfaced in API/UI |
| Dev split 3/1/1 years vs full 15/2/3 | Proportional scaling for the 5-year dev subset; see `config.yaml` |
| `scripts/tasks.py` behind the Makefile | Windows machines often lack `make` |

Detail for each module is added as its milestone lands.
