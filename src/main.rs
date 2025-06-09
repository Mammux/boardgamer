mod tic_tac_toe;
use tic_tac_toe::*;
use plotters::prelude::*;
use std::env;

// This program trains a neural tic-tac-toe player through repeated self-play.
// After each generation of training the new player is evaluated against the
// strongest earlier generation and the results are visualised as a heatmap.


const NUM_GENERATIONS: usize = 100;
const TRAIN_GAMES: usize = 1000;
const EVAL_GAMES: usize = 10;

/// Play a single self-play game between two networks and train them according
/// to the final outcome. The return value is the score from the perspective of
/// `p1` (1 = win, -1 = loss, 0 = draw).
fn play_game(p1: &mut NeuralPlayer, p2: &mut NeuralPlayer) -> i32 {
    let mut board = Board::new();
    let mut states_p1 = Vec::new();
    let mut actions_p1 = Vec::new();
    let mut states_p2 = Vec::new();
    let mut actions_p2 = Vec::new();

    loop {
        // Player 1 turn
        // Capture the board state before making a move so the network learns
        // which action to take given the current situation. Previously the
        // state was recorded after the move, causing the network to learn to
        // repeat moves on already occupied cells.
        let state = board_to_state(&board, 1);
        let action = p1.select_action(&board, 1);
        board.make_move(action);
        states_p1.push(state);
        actions_p1.push(action);
        match board.check_winner() {
            GameResult::Win(w) => {
                if w == 1 { p1.train(&states_p1, &actions_p1, 1.0); p2.train(&states_p2, &actions_p2, -1.0); return 1; }
                else { p1.train(&states_p1, &actions_p1, -1.0); p2.train(&states_p2, &actions_p2, 1.0); return -1; }
            },
            GameResult::Draw => {
                p1.train(&states_p1, &actions_p1, 0.0);
                p2.train(&states_p2, &actions_p2, 0.0);
                return 0;
            },
            GameResult::Ongoing => {}
        }

        // Player 2 turn
        let state = board_to_state(&board, -1);
        let action = p2.select_action(&board, -1);
        board.make_move(action);
        states_p2.push(state);
        actions_p2.push(action);
        match board.check_winner() {
            GameResult::Win(w) => {
                if w == 1 { p1.train(&states_p1, &actions_p1, 1.0); p2.train(&states_p2, &actions_p2, -1.0); return 1; }
                else { p1.train(&states_p1, &actions_p1, -1.0); p2.train(&states_p2, &actions_p2, 1.0); return -1; }
            },
            GameResult::Draw => {
                p1.train(&states_p1, &actions_p1, 0.0);
                p2.train(&states_p2, &actions_p2, 0.0);
                return 0;
            },
            GameResult::Ongoing => {}
        }
    }
}

fn idx_to_coord(idx: usize) -> &'static str {
    match idx {
        0 => "a1",
        1 => "b1",
        2 => "c1",
        3 => "a2",
        4 => "b2",
        5 => "c2",
        6 => "a3",
        7 => "b3",
        8 => "c3",
        _ => "?",
    }
}

/// Run a single game between two copies of the provided player and print each
/// move to stdout. Useful for observing how a trained model behaves.
fn play_game_log(player: &mut NeuralPlayer) {
    let mut board = Board::new();
    let mut p1 = player.clone();
    let mut p2 = player.clone();
    p1.lr = 0.0;
    p2.lr = 0.0;

    loop {
        let (act, symbol) = if board.player == 1 {
            (p1.select_action(&board, 1), 'X')
        } else {
            (p2.select_action(&board, -1), 'O')
        };
        board.make_move(act);
        println!("{} {}", symbol, idx_to_coord(act));
        match board.check_winner() {
            GameResult::Win(w) => {
                println!("{} wins", if w == 1 { 'X' } else { 'O' });
                break;
            }
            GameResult::Draw => {
                println!("Draw");
                break;
            }
            GameResult::Ongoing => {}
        }
    }
}

/// Train the network over multiple generations and optionally play out a
/// demonstration game with the final model when `--final-game` is passed.
fn main() {
    let play_final = env::args().any(|a| a == "--final-game");

    let mut generations = Vec::new();
    let mut player = NeuralPlayer::new(0, 0.01);

    for gen in 0..NUM_GENERATIONS {
        // player.save(&format!("models/gen_{}.bin", gen)).unwrap();
        generations.push(player.clone());

        if gen < NUM_GENERATIONS - 1 {
            // Determine the strongest previous generation by evaluating each
            // against the current player. The one with the highest score over a
            // small evaluation set is selected as the training opponent.
            // Evaluate previous generations and pick the best one as the
            // opponent for further training.
            let strongest_idx = if generations.len() > 1 {
                let mut best_idx = 0;
                let mut best_score = i32::MIN;
                for idx in 0..generations.len() - 1 {
                    let mut opp = generations[idx].clone();
                    let mut me = player.clone();
                    opp.lr = 0.0;
                    me.lr = 0.0;
                    let mut score = 0i32;
                    for g in 0..EVAL_GAMES {
                        if g % 2 == 0 {
                            score += play_game(&mut opp, &mut me);
                        } else {
                            score -= play_game(&mut me, &mut opp);
                        }
                    }
                    if score > best_score {
                        best_score = score;
                        best_idx = idx;
                    }
                }
                best_idx
            } else {
                0
            };

            // Train against the selected opponent for a number of games.
            for i in 0..TRAIN_GAMES {
                let mut opponent = generations[strongest_idx].clone();
                opponent.lr = 0.0;
                if i % 2 == 0 {
                    play_game(&mut player, &mut opponent);
                } else {
                    play_game(&mut opponent, &mut player);
                }
            }
        }
    }

    // Build a matrix of win rates between all generations for visualisation.
    let mut matrix = vec![vec![0f32; NUM_GENERATIONS]; NUM_GENERATIONS];
    for i in 0..NUM_GENERATIONS {
        for j in 0..NUM_GENERATIONS {
            if i == j { continue; }
            let mut p1 = generations[i].clone();
            let mut p2 = generations[j].clone();
            p1.lr = 0.0;
            p2.lr = 0.0;
            let mut score = 0i32;
            for g in 0..EVAL_GAMES {
                if g % 2 == 0 {
                    score += play_game(&mut p1, &mut p2);
                } else {
                    score -= play_game(&mut p2, &mut p1);
                }
            }
            matrix[i][j] = score as f32 / EVAL_GAMES as f32;
        }
    }

    let root = BitMapBackend::new("generation_strength.png", (800, 800)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .caption("Generation Strength", ("sans-serif", 30))
        .margin(20)
        .set_all_label_area_size(40)
        .build_cartesian_2d(0..NUM_GENERATIONS as i32, 0..NUM_GENERATIONS as i32)
        .unwrap();

    chart.configure_mesh().disable_mesh().draw().unwrap();

    for i in 0..NUM_GENERATIONS {
        for j in 0..NUM_GENERATIONS {
            let val = matrix[i][j];
            let hue = 240.0 - ((val + 1.0) / 2.0 * 240.0); // -1 -> blue, 1 -> red
            let color = HSLColor((hue / 360.0) as f64, 1.0, 0.5);
            chart
                .draw_series(std::iter::once(Rectangle::new(
                    [(i as i32, j as i32), (i as i32 + 1, j as i32 + 1)],
                    ShapeStyle::from(&color).filled(),
                )))
                .unwrap();
        }
    }

    if play_final {
        println!("Final generation self-play:");
        let mut last = generations.last().unwrap().clone();
        play_game_log(&mut last);
    }
}
