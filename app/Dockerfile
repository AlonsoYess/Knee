# Build context: app/ only, or the explicit TAR from deploy/build_context.py.
# Official image indexes verified in the registry. Update pins deliberately and retest.
FROM node:22.17.0-bookworm-slim@sha256:b04ce4ae4e95b522112c2e5c52f781471a5cbc3b594527bcddedee9bc48c03a0 AS frontend-build
WORKDIR /build/frontend
ENV CI=true
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --ignore-scripts --no-audit --no-fund
COPY frontend/index.html frontend/tsconfig.json frontend/vite.config.ts ./
COPY frontend/src/ ./src/
COPY frontend/public/manifest.webmanifest ./public/manifest.webmanifest
COPY frontend/public/icons/ ./public/icons/
COPY frontend/scripts/pwa-build.mjs frontend/scripts/pwa-build.d.mts frontend/scripts/pwa-worker.js ./scripts/
ARG KNEEOAI_BUILD_ID
RUN KNEEOAI_BUILD_ID="${KNEEOAI_BUILD_ID}" npm run build

FROM python:3.12.14-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e AS python-dependencies
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /build
COPY backend/requirements.lock ./requirements.lock
RUN python -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --only-binary=:all: --no-deps -r requirements.lock \
    && /opt/venv/bin/python -m pip check

FROM python:3.12.14-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e AS application
LABEL org.opencontainers.image.title="KneeOAI application" \
      org.opencontainers.image.description="Application-only prototype; no scientific weights, cohorts or predictor included."
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    KNEEOAI_ENV=production \
    KNEEOAI_DATA_DIR=/var/lib/kneeoai \
    KNEEOAI_FRONTEND_DIST=/opt/kneeoai/frontend \
    PORT=8080
RUN groupadd --gid 10001 kneeoai \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin kneeoai \
    && install -d -m 0755 /opt/kneeoai \
    && install -d -m 0750 -o 10001 -g 10001 /var/lib/kneeoai
WORKDIR /opt/kneeoai
COPY --from=python-dependencies /opt/venv /opt/venv
COPY --from=frontend-build /build/frontend/dist/ ./frontend/
COPY backend/knee_app/ ./knee_app/
COPY backend/requirements.lock ./requirements.lock
USER 10001:10001
EXPOSE 8080
# Always attach an explicit persistent volume; the image contains no case database.
VOLUME ["/var/lib/kneeoai"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import json,os,sys,urllib.request; r=urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8080')+'/api/v1/health',timeout=3); sys.exit(0 if r.status==200 and json.load(r).get('status')=='ok' else 1)"]
ENTRYPOINT ["python", "-m", "knee_app.deploy"]
