# ChemX Deployment

## Architecture

```
Streamlit Community Cloud
        |
        | HTTPS JSON
        v
Render Web Service
  FastAPI + model bundle
        |
        +--> PLS density model
        +--> Bayesian uncertainty
        +--> PCA/MSPC monitoring
        +--> CSTR simulation/optimization
```

The Streamlit frontend does not require the raw public datasets. The backend ships with
the small versioned MLNIR model bundle under `models/`. Raw datasets remain local-only
for reproducibility and offline retraining.

## Render backend

The repository includes `render.yaml` and a backend Dockerfile.

1. Create/sync the Render Blueprint from the repository.
2. Set `CHEMX_CORS_ORIGINS` to the exact deployed Streamlit origin.
3. Set `GROQ_API_KEY` only if stakeholder LLM narratives are desired.
4. Render health check: `GET /health`.
5. Readiness: `GET /ready`.
6. OpenAPI: `GET /docs`.

Render supplies the public `PORT`; the Docker entrypoint binds Uvicorn to it.

## Streamlit frontend

Deploy `src/dashboard/app.py` as the Streamlit Community Cloud entrypoint.

Set the secret:

```toml
CHEMX_API_URL = "https://<your-render-service>.onrender.com"
```

The frontend uses `requirements-streamlit.txt`. The API key for Groq is never needed
in the browser-facing app; it remains a backend secret.

## Local development

Terminal 1:

```bash
uvicorn src.api.main:app --reload --port 8000
```

Terminal 2:

```set CHEMX_API_URL=http://localhost:8000
streamlit run src/dashboard/app.py
```

## Deployment boundary

- MLNIR, Corn, and Sugar are public experimental datasets.
- The deployed density model is trained on MLNIRdata.
- CSTR physics and optimization are `MECHANISTIC_SIMULATION` only.
- No plant-control recommendation, Shell data, refinery data, LNG data, or proprietary
  process data is represented.
