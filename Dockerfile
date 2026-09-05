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
# BACKEND_SUMMARY.md. --workers 1 keeps memory low enough for the free-tier
# 256MB VM in fly.toml — bump both if you outgrow it.
#
# --worker-class gthread --threads 4 matters independently of --timeout:
# the default sync worker class handles exactly one request at a time, full
# stop — a slow /api/humanize (blocked on Modal) makes the ENTIRE server
# unresponsive to everything else, including a page refresh or another
# visitor's request, until that one call finishes. ui/app.py's
# `threaded=True` on app.run() only affects `python app.py` directly — it
# does nothing here, since gunicorn imports the app as a WSGI callable and
# never executes that `if __name__ == "__main__"` block at all. gthread is
# the right choice (not sync, not sync+more workers) because this workload
# is I/O-bound — waiting on Modal HTTP responses, not burning CPU — so
# threads share the 256MB memory budget instead of multiplying it the way
# more worker processes would.
CMD ["gunicorn", "ui.app:app", "--bind", "0.0.0.0:8080", "--workers", "1", "--worker-class", "gthread", "--threads", "4", "--timeout", "180"]
