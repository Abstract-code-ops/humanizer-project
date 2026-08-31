FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ui/app.py does `sys.path.insert(0, ROOT)` and imports config.py,
# humanizers/, detectors/, evaluation/ from the project root — so the whole
# backend needs to ship, not just ui/. data/ is included for results.csv
# (Compare page) and data/source_texts/ (sample texts).
COPY config.py config.yaml ./
COPY humanizers/ ./humanizers/
COPY detectors/ ./detectors/
COPY evaluation/ ./evaluation/
COPY data/ ./data/
COPY ui/ ./ui/

EXPOSE 8080

# --timeout 180 matches the worst-case Modal cold-start window documented in
# BACKEND_SUMMARY.md (paraphrase + backtranslate + detect can all be cold at
# once on the first /api/humanize call). --workers 1 keeps memory low enough
# for the free-tier 256MB VM in fly.toml — bump both if you outgrow it.
CMD ["gunicorn", "ui.app:app", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "180"]
