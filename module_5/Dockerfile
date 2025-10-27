FROM agrigorev/zoomcamp-model:2025

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"

RUN mv /code/pipeline_v2.bin /app/

COPY "pyproject.toml" "uv.lock" ".python-version" ./
RUN uv sync --locked

# COPY "app.py" "pipeline_v1.bin" ./
COPY "app.py" ./

EXPOSE 8000

ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]