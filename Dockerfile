# Start from an official slim Python image — gives us Python 3.11 on Debian
FROM python:3.11-slim

# Install uv (the package manager this project uses)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set the working directory inside the container
WORKDIR /app

# Copy dependency files first — Docker caches layers, so if these don't change,
# the expensive "install packages" step is skipped on rebuilds
COPY pyproject.toml uv.lock ./

# Install dependencies into the system Python (no virtualenv needed inside Docker)
RUN uv sync --frozen --no-dev --no-install-project

# Copy the rest of the project
COPY src/ ./src/
COPY api/ ./api/
COPY index.html ./
COPY companies.yaml ./
COPY scripts/ ./scripts/

# Tell Docker this container listens on port 8000
EXPOSE 8000

# The command to run when the container starts
CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
