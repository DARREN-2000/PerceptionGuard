# Contributing to Inference Control Plane

First off, thank you for considering contributing to Inference Control Plane! It's people like you that make open-source software great.

Inference Control Plane aims to be the standard open-source Inference Control Plane. We welcome contributions from everyone, whether it's fixing bugs, adding new features, improving documentation, or optimizing performance.

## Getting Started

1. **Read the Docs:** Familiarize yourself with the architecture and core concepts by reading the `docs/` folder.
2. **Find an Issue:** Look for issues tagged `good first issue` or `help wanted` in the GitHub issue tracker.
3. **Discuss Before Building:** If you plan to add a major feature, please open an issue to discuss the design with the maintainers before writing code. This saves everyone time!

## Local Development Setup

Please refer to the [Installation Guide](docs/installation.md) and [Developer Experience Guide](docs/developer-experience.md) for detailed instructions on setting up the Python backend and Next.js frontend.

## Pull Request Process

1. **Fork the Repo:** Create a fork and clone it locally.
2. **Branch Naming:** Create a branch for your feature or bug fix (e.g., `feature/semantic-cache` or `fix/token-calculation`).
3. **Write Code & Tests:**
   - Ensure your code follows the existing style.
   - For backend changes, add asynchronous unit tests in `tests/`.
   - Run the linting tools before committing: `make lint-backend`.
4. **Submit a PR:**
   - Fill out the provided Pull Request Template completely.
   - Link any relevant issues.
   - Wait for CI checks (GitHub Actions) to pass.
5. **Code Review:** A maintainer will review your code. We may request changes. Please be responsive!

## Coding Standards

### Backend (Python)
- We use **FastAPI** and strict **asyncio**. Do not introduce blocking I/O calls.
- Use **Pydantic** for all data validation.
- Format code using `ruff` (`uv run ruff format src/ tests/`).
- Use Type Hints (`-> list[str]`, etc.) everywhere.

### Frontend (Next.js/React)
- Use **Tailwind CSS v4** exclusively for styling.
- Use **pnpm** (not npm or yarn).
- Ensure the build succeeds (`pnpm run build`) without TypeScript errors.

## Community

If you have questions, please start a [GitHub Discussion](https://github.com/DARREN-2000/perceptionguard/discussions).

Thank you for contributing!
