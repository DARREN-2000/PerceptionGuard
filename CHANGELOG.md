# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive enterprise-grade documentation suite.
- GitHub community templates (Issues, PRs).
- New `CODEOWNERS` and `SECURITY.md` policies.

### Changed
- Refactored `README.md` to highlight enterprise capabilities and deployment options.

## [1.0.0] - Initial Release

### Added
- Core Proxy API built with FastAPI and asyncpg.
- Redis-based exact caching and rate-limiting.
- Multi-provider fallback routing logic (OpenAI, Anthropic, Azure).
- PostgreSQL integration for persistent usage logging.
- Next.js Admin Dashboard for tenant and key management.
- Docker Compose stack for rapid local deployment.
- Base Kubernetes manifests for production deployment.
- Prometheus metrics endpoint (`/metrics`).
