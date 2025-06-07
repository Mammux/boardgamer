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

const HIDDEN_SIZE: usize = 128;

#[derive(Clone)]
pub struct NeuralPlayer {
    pub w1: [[f32; BOARD_SIZE]; HIDDEN_SIZE],   // input -> hidden
    pub b1: [f32; HIDDEN_SIZE],
    pub w2: [[f32; HIDDEN_SIZE]; BOARD_SIZE],   // hidden -> output
    pub b2: [f32; BOARD_SIZE],
    pub lr: f32,
    pub rng: StdRng,
}

impl NeuralPlayer {
    pub fn new(seed: u64, lr: f32) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut w1 = [[0.0; BOARD_SIZE]; HIDDEN_SIZE];
        let mut b1 = [0.0; HIDDEN_SIZE];
        let mut w2 = [[0.0; HIDDEN_SIZE]; BOARD_SIZE];
        let mut b2 = [0.0; BOARD_SIZE];
        // Small random init
        for i in 0..HIDDEN_SIZE {
            for j in 0..BOARD_SIZE {
                w1[i][j] = (rng.gen::<f32>() - 0.5) * 0.2;
            }
            b1[i] = 0.0;
        }
        for i in 0..BOARD_SIZE {
            for j in 0..HIDDEN_SIZE {
                w2[i][j] = (rng.gen::<f32>() - 0.5) * 0.2;
            }
            b2[i] = 0.0;
        }
        Self { w1, b1, w2, b2, lr, rng }
    }

    pub fn select_action(&mut self, board: &Board, perspective: i32) -> usize {
        let state = board_to_state(board, perspective);
        let logits = self.forward(&state);
        let mut probs = softmax(&logits);

        // Mask out invalid moves so the player does not try placing a piece on
        // an already occupied cell. This prevents the training from getting
        // stuck in states where the network repeatedly selects illegal actions
        // and ensures learning focuses on meaningful gameplay.
        for i in 0..BOARD_SIZE {
            if !board.is_valid_move(i) {
                probs[i] = 0.0;
            }
        }

        // Re-normalise the probabilities. If all moves are invalid (which
        // should only happen once the board is full), fall back to choosing a
        // random valid move.
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for p in probs.iter_mut() {
                *p /= sum;
            }
            sample_from_probs(&mut self.rng, &probs)
        } else {
            let valid_moves: Vec<usize> = (0..BOARD_SIZE)
                .filter(|&i| board.is_valid_move(i))
                .collect();
            if valid_moves.is_empty() {
                0
            } else {
                let idx = self.rng.gen_range(0..valid_moves.len());
                valid_moves[idx]
            }
        }
    }

    fn forward(&self, state: &[f32; BOARD_SIZE]) -> [f32; BOARD_SIZE] {
        // Input -> hidden
        let mut hidden = [0f32; HIDDEN_SIZE];
        for i in 0..HIDDEN_SIZE {
            for j in 0..BOARD_SIZE {
                hidden[i] += self.w1[i][j] * state[j];
            }
            hidden[i] += self.b1[i];
            // ReLU
            if hidden[i] < 0.0 { hidden[i] = 0.0; }
        }
        // Hidden -> output
        let mut out = [0f32; BOARD_SIZE];
        for i in 0..BOARD_SIZE {
            for j in 0..HIDDEN_SIZE {
                out[i] += self.w2[i][j] * hidden[j];
            }
            out[i] += self.b2[i];
        }
        out
    }

    pub fn train(&mut self, states: &[[f32; BOARD_SIZE]], actions: &[usize], reward: f32) {
        const DISCOUNT: f32 = 0.9;
        let n = states.len();

        // Temporary gradient accumulators
        let mut dw1 = [[0.0; BOARD_SIZE]; HIDDEN_SIZE];
        let mut db1 = [0.0; HIDDEN_SIZE];
        let mut dw2 = [[0.0; HIDDEN_SIZE]; BOARD_SIZE];
        let mut db2 = [0.0; BOARD_SIZE];

        for (idx, (state, &action)) in states.iter().zip(actions.iter()).enumerate() {
            let disc = DISCOUNT.powi((n - idx - 1) as i32);
            // Forward
            let mut hidden = [0f32; HIDDEN_SIZE];
            let mut hidden_raw = [0f32; HIDDEN_SIZE];
            for i in 0..HIDDEN_SIZE {
                for j in 0..BOARD_SIZE {
                    hidden_raw[i] += self.w1[i][j] * state[j];
                }
                hidden_raw[i] += self.b1[i];
                hidden[i] = hidden_raw[i].max(0.0); // ReLU
            }
            let logits = {
                let mut out = [0f32; BOARD_SIZE];
                for i in 0..BOARD_SIZE {
                    for j in 0..HIDDEN_SIZE {
                        out[i] += self.w2[i][j] * hidden[j];
                    }
                    out[i] += self.b2[i];
                }
                out
            };
            let probs = softmax(&logits);

            // Output layer gradients
            let mut dlogits = [0f32; BOARD_SIZE];
            for i in 0..BOARD_SIZE {
                dlogits[i] = (if i == action { 1.0 } else { 0.0 }) - probs[i];
            }

            // Gradients for w2, b2, and hidden
            let mut dhidden = [0f32; HIDDEN_SIZE];
            for i in 0..BOARD_SIZE {
                for j in 0..HIDDEN_SIZE {
                    dw2[i][j] += reward * disc * dlogits[i] * hidden[j];
                    dhidden[j] += self.w2[i][j] * dlogits[i];
                }
                db2[i] += reward * disc * dlogits[i];
            }

            // Gradients for w1, b1
            for i in 0..HIDDEN_SIZE {
                let grad = if hidden_raw[i] > 0.0 { dhidden[i] } else { 0.0 };
                for j in 0..BOARD_SIZE {
                    dw1[i][j] += reward * disc * grad * state[j];
                }
                db1[i] += reward * disc * grad;
            }
        }

        // Apply accumulated gradients
        for i in 0..HIDDEN_SIZE {
            for j in 0..BOARD_SIZE {
                self.w1[i][j] += self.lr * dw1[i][j];
            }
            self.b1[i] += self.lr * db1[i];
        }
        for i in 0..BOARD_SIZE {
            for j in 0..HIDDEN_SIZE {
                self.w2[i][j] += self.lr * dw2[i][j];
            }
            self.b2[i] += self.lr * db2[i];
        }
    }

    /*
    // Update save/load to include new weights
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let data = bincode::serialize(&(self.w1, self.b1, self.w2, self.b2)).unwrap();
        std::fs::write(path, data)
    }

    pub fn _load(path: &str, seed: u64, lr: f32) -> std::io::Result<Self> {
        let data = std::fs::read(path)?;
        let (w1, b1, w2, b2): ([[f32; BOARD_SIZE]; HIDDEN_SIZE], [f32; HIDDEN_SIZE], [[f32; HIDDEN_SIZE]; BOARD_SIZE], [f32; BOARD_SIZE]) =
            bincode::deserialize(&data).unwrap();
        Ok(Self { w1, b1, w2, b2, lr, rng: StdRng::seed_from_u64(seed) })
    }
    */
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_move_and_player_switch() {
        let mut board = Board::new();
        assert!(board.make_move(0));
        // player should switch from 1 to -1
        assert_eq!(board.player, -1);
        // cell updated
        assert_eq!(board.cells[0], 1);
        // cannot move to same cell again
        assert!(!board.make_move(0));
    }

    #[test]
    fn test_board_to_state_perspective() {
        let mut board = Board::new();
        board.make_move(0); // player 1 moves
        board.make_move(1); // player -1 moves
        let state_p1 = board_to_state(&board, 1);
        let state_p2 = board_to_state(&board, -1);
        // board cells: [1, -1, 0, ...]
        assert_eq!(state_p1[0], 1.0);
        assert_eq!(state_p1[1], -1.0);
        assert_eq!(state_p2[0], -1.0); // from opponent perspective
        assert_eq!(state_p2[1], 1.0);
    }

    #[test]
    fn test_select_action_returns_valid_move() {
        let mut player = NeuralPlayer::new(0, 0.0);
        let mut board = Board::new();
        board.make_move(0); // occupy cell 0
        for _ in 0..10 {
            let action = player.select_action(&board, 1);
            assert!(board.is_valid_move(action));
        }
    }

    fn play_game_train(p1: &mut NeuralPlayer, p2: &mut NeuralPlayer) -> i32 {
        let mut board = Board::new();
        let mut states_p1 = Vec::new();
        let mut actions_p1 = Vec::new();
        let mut states_p2 = Vec::new();
        let mut actions_p2 = Vec::new();

        loop {
            let state = board_to_state(&board, 1);
            let action = p1.select_action(&board, 1);
            board.make_move(action);
            states_p1.push(state);
            actions_p1.push(action);
            match board.check_winner() {
                GameResult::Win(w) => {
                    if w == 1 {
                        p1.train(&states_p1, &actions_p1, 1.0);
                        p2.train(&states_p2, &actions_p2, -1.0);
                        return 1;
                    } else {
                        p1.train(&states_p1, &actions_p1, -1.0);
                        p2.train(&states_p2, &actions_p2, 1.0);
                        return -1;
                    }
                }
                GameResult::Draw => {
                    p1.train(&states_p1, &actions_p1, 0.0);
                    p2.train(&states_p2, &actions_p2, 0.0);
                    return 0;
                }
                GameResult::Ongoing => {}
            }

            let state = board_to_state(&board, -1);
            let action = p2.select_action(&board, -1);
            board.make_move(action);
            states_p2.push(state);
            actions_p2.push(action);
            match board.check_winner() {
                GameResult::Win(w) => {
                    if w == 1 {
                        p1.train(&states_p1, &actions_p1, 1.0);
                        p2.train(&states_p2, &actions_p2, -1.0);
                        return 1;
                    } else {
                        p1.train(&states_p1, &actions_p1, -1.0);
                        p2.train(&states_p2, &actions_p2, 1.0);
                        return -1;
                    }
                }
                GameResult::Draw => {
                    p1.train(&states_p1, &actions_p1, 0.0);
                    p2.train(&states_p2, &actions_p2, 0.0);
                    return 0;
                }
                GameResult::Ongoing => {}
            }
        }
    }

    fn play_game_eval(p1: &mut NeuralPlayer, p2: &mut NeuralPlayer) -> i32 {
        let mut board = Board::new();
        loop {
            let action = p1.select_action(&board, 1);
            board.make_move(action);
            match board.check_winner() {
                GameResult::Win(w) => return if w == 1 { 1 } else { -1 },
                GameResult::Draw => return 0,
                GameResult::Ongoing => {}
            }

            let action = p2.select_action(&board, -1);
            board.make_move(action);
            match board.check_winner() {
                GameResult::Win(w) => return if w == 1 { 1 } else { -1 },
                GameResult::Draw => return 0,
                GameResult::Ongoing => {}
            }
        }
    }

    #[test]
    fn test_trained_model_beats_random() {
        let mut trained = NeuralPlayer::new(0, 0.01);
        let mut opponent = NeuralPlayer::new(1, 0.0);

        // Train the model for a modest number of games
        for i in 0..200 {
            if i % 2 == 0 {
                play_game_train(&mut trained, &mut opponent);
            } else {
                play_game_train(&mut opponent, &mut trained);
            }
        }

        let mut trained_eval = trained.clone();
        trained_eval.lr = 0.0;
        let mut random_eval = NeuralPlayer::new(2, 0.0);

        let mut score_trained = 0i32;
        for i in 0..100 {
            if i % 2 == 0 {
                score_trained += play_game_eval(&mut trained_eval, &mut random_eval);
            } else {
                score_trained -= play_game_eval(&mut random_eval, &mut trained_eval);
            }
        }

        let mut baseline_p1 = NeuralPlayer::new(3, 0.0);
        let mut baseline_p2 = NeuralPlayer::new(4, 0.0);
        let mut score_baseline = 0i32;
        for i in 0..100 {
            if i % 2 == 0 {
                score_baseline += play_game_eval(&mut baseline_p1, &mut baseline_p2);
            } else {
                score_baseline -= play_game_eval(&mut baseline_p2, &mut baseline_p1);
            }
        }

        assert!(score_trained > score_baseline, "trained: {}, baseline: {}", score_trained, score_baseline);
    }
}
