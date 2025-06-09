# Agent Guidelines

This repository is a small Rust project that trains a neural network to play tic-tac-toe. The primary files are:

- `src/main.rs` – orchestrates training across generations and produces a heatmap of win rates.
- `src/tic_tac_toe.rs` – contains the game logic, neural network implementation and tests.

## Building

Use the standard Rust tooling. The project has no custom build steps.

```bash
cargo build --release     # compile the project
cargo run --release -- [--final-game]  # run with optional demo game
```

## Testing

Always run the tests and a type check when modifying the code:

```bash
cargo test --quiet
cargo check --quiet
```

These commands should complete without warnings or failures before committing.
