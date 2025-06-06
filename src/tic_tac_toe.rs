pub const BOARD_SIZE: usize = 9;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GameResult {
    Win(i32),
    Draw,
    Ongoing,
}

#[derive(Clone)]
pub struct Board {
    pub cells: [i32; BOARD_SIZE],
    pub player: i32,
}

impl Board {
    pub fn new() -> Self {
        Self { cells: [0; BOARD_SIZE], player: 1 }
    }

    pub fn is_valid_move(&self, idx: usize) -> bool {
        idx < BOARD_SIZE && self.cells[idx] == 0
    }

    pub fn make_move(&mut self, idx: usize) -> bool {
        if self.is_valid_move(idx) {
            self.cells[idx] = self.player;
            self.player *= -1;
            true
        } else {
            false
        }
    }

    pub fn check_winner(&self) -> GameResult {
        const WIN_LINES: [[usize; 3]; 8] = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
        ];

        for line in WIN_LINES.iter() {
            let sum = self.cells[line[0]] + self.cells[line[1]] + self.cells[line[2]];
            if sum == 3 {
                return GameResult::Win(1);
            } else if sum == -3 {
                return GameResult::Win(-1);
            }
        }

        if self.cells.iter().all(|&c| c != 0) {
            GameResult::Draw
        } else {
            GameResult::Ongoing
        }
    }
}

pub fn board_to_state(board: &Board, perspective: i32) -> [f32; BOARD_SIZE] {
    let mut state = [0f32; BOARD_SIZE];
    for i in 0..BOARD_SIZE {
        state[i] = board.cells[i] as f32 * perspective as f32;
    }
    state
}

use rand::{rngs::StdRng, Rng, SeedableRng};

#[derive(Clone)]
pub struct NeuralPlayer {
    pub weights: [[f32; BOARD_SIZE]; BOARD_SIZE],
    pub lr: f32,
    pub rng: StdRng,
}

impl NeuralPlayer {
    pub fn new(seed: u64, lr: f32) -> Self {
        Self { weights: [[0.0; BOARD_SIZE]; BOARD_SIZE], lr, rng: StdRng::seed_from_u64(seed) }
    }

    pub fn select_action(&mut self, board: &Board, perspective: i32) -> usize {
        let state = board_to_state(board, perspective);
        let logits = self.forward(&state);
        let probs = softmax(&logits);
        sample_from_probs(&mut self.rng, &probs)
    }

    fn forward(&self, state: &[f32; BOARD_SIZE]) -> [f32; BOARD_SIZE] {
        let mut out = [0f32; BOARD_SIZE];
        for i in 0..BOARD_SIZE {
            for j in 0..BOARD_SIZE {
                out[i] += self.weights[i][j] * state[j];
            }
        }
        out
    }

    pub fn train(&mut self, states: &[[f32; BOARD_SIZE]], actions: &[usize], reward: f32) {
        for (state, &action) in states.iter().zip(actions.iter()) {
            let logits = self.forward(state);
            let probs = softmax(&logits);
            for i in 0..BOARD_SIZE {
                let grad = (if i == action { 1.0 } else { 0.0 }) - probs[i];
                for j in 0..BOARD_SIZE {
                    self.weights[i][j] += self.lr * reward * grad * state[j];
                }
            }
        }
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let data = bincode::serialize(&self.weights).unwrap();
        std::fs::write(path, data)
    }

    pub fn load(path: &str, seed: u64, lr: f32) -> std::io::Result<Self> {
        let data = std::fs::read(path)?;
        let weights: [[f32; BOARD_SIZE]; BOARD_SIZE] = bincode::deserialize(&data).unwrap();
        Ok(Self { weights, lr, rng: StdRng::seed_from_u64(seed) })
    }
}

fn softmax(logits: &[f32; BOARD_SIZE]) -> [f32; BOARD_SIZE] {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exp: [f32; BOARD_SIZE] = [0.0; BOARD_SIZE];
    let mut sum = 0.0;
    for i in 0..BOARD_SIZE {
        exp[i] = (logits[i] - max).exp();
        sum += exp[i];
    }
    for i in 0..BOARD_SIZE {
        exp[i] /= sum;
    }
    exp
}

fn sample_from_probs(rng: &mut StdRng, probs: &[f32; BOARD_SIZE]) -> usize {
    let mut r = rng.gen::<f32>();
    for i in 0..BOARD_SIZE {
        if r < probs[i] { return i; }
        r -= probs[i];
    }
    BOARD_SIZE - 1
}
