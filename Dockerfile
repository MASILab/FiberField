# From https://github.com/astral-sh/uv-docker-example
# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim

# Add requirements for VTK
RUN apt-get update && apt-get install -y --no-install-recommends \
    tini \
    xvfb \
    xauth \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxt6 \
    libgl1 \
    libglu1-mesa \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Setup a non-root user
# RUN groupadd --system --gid 999 nonroot \
#     && useradd --system --gid 999 --uid 999 --create-home nonroot

# Install the project into `/app`
WORKDIR /app

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Omit development dependencies
ENV UV_NO_DEV=1

# Ensure installed tools can be executed out of the box
ENV UV_TOOL_BIN_DIR=/usr/local/bin

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Entrypoint needs to be tini so that xvfb-run doesn't hang
# We need xvfb-run to run GUI applications in a headless environment
ENV TINI_SUBREAPER=1
ENTRYPOINT ["tini", "--", "xvfb-run", "-a"]

# Use the non-root user to run our application
# USER nonroot

CMD ["fiberfield"]