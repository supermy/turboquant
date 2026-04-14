.PHONY: help build test test-verbose test-integration bench clean lint doc check-all quick-check run-example run-example-release

CARGO := cargo
TARGET_DIR := target

help:
	@echo "PolarQuant (Rust) - Lossless Quantization via Polar Transformation"
	@echo ""
	@echo "Available targets:"
	@echo "  build              - Build the library (debug)"
	@echo "  build-release      - Build the library (release/optimized)"
	@echo "  test               - Run all tests"
	@echo "  test-verbose       - Run tests with verbose output"
	@echo "  test-integration   - Run integration tests only"
	@echo "  bench              - Run performance benchmarks"
	@echo "  lint               - Run clippy linter"
	@echo "  fmt                - Format code with rustfmt"
	@echo "  fmt-check          - Check code formatting"
	@echo "  doc                - Generate documentation"
	@echo "  doc-open           - Generate and open documentation"
	@echo "  clean              - Clean build artifacts"
	@echo "  check-all          - Run all checks (fmt, lint, test)"
	@echo "  quick-check        - Run quick unit tests only"
	@echo "  run-example        - Run the KV cache demo (debug)"
	@echo "  run-example-release- Run the KV cache demo (release/optimized)"

build:
	$(CARGO) build

build-release:
	$(CARGO) build --release

test:
	$(CARGO) test

test-verbose:
	$(CARGO) test -- --nocapture

test-integration:
	$(CARGO) test --test integration

test-unit:
	$(CARGO) test --lib

bench:
	$(CARGO) bench

lint:
	$(CARGO) clippy -- -D warnings

fmt:
	$(CARGO) fmt

fmt-check:
	$(CARGO) fmt -- --check

doc:
	$(CARGO) doc --no-deps

doc-open:
	$(CARGO) doc --no-deps --open

clean:
	$(CARGO) clean
	rm -rf $(TARGET_DIR)
	rm -rf criterion

check-all: fmt-check lint test
	@echo "All checks passed!"

quick-check: test-unit
	@echo "Quick checks passed!"

run-example:
	$(CARGO) run --example kv_cache_demo

run-example-release:
	$(CARGO) run --example kv_cache_demo --release
