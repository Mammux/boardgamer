mod tic_tac_toe;
use tic_tac_toe::*;
use plotters::prelude::*;
use std::fs::create_dir_all;

const NUM_GENERATIONS: usize = 20;
const TRAIN_GAMES: usize = 1000;
const EVAL_GAMES: usize = 100;

fn play_game(p1: &mut NeuralPlayer, p2: &mut NeuralPlayer) -> i32 {
    let mut board = Board::new();
    let mut states_p1 = Vec::new();
    let mut actions_p1 = Vec::new();
    let mut states_p2 = Vec::new();
    let mut actions_p2 = Vec::new();

    loop {
        // Player 1 turn
        let action = p1.select_action(&board, 1);
        board.make_move(action);
        states_p1.push(board_to_state(&board, 1));
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
        let action = p2.select_action(&board, -1);
        board.make_move(action);
        states_p2.push(board_to_state(&board, -1));
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

fn main() {
    create_dir_all("models").unwrap();

    let mut generations = Vec::new();
    let mut player = NeuralPlayer::new(0, 0.01);

    for gen in 0..NUM_GENERATIONS {
        // Save the current generation before further training
        player.save(&format!("models/gen_{}.bin", gen)).unwrap();
        generations.push(player.clone());

        // Train the model for the next generation using self-play
        if gen < NUM_GENERATIONS - 1 {
            for i in 0..TRAIN_GAMES {
                let mut opponent = player.clone();
                opponent.lr = 0.0;
                if i % 2 == 0 {
                    play_game(&mut player, &mut opponent);
                } else {
                    play_game(&mut opponent, &mut player);
                }
            }
        }
    }

    let mut matrix = vec![vec![0f32; NUM_GENERATIONS]; NUM_GENERATIONS];
    for i in 0..NUM_GENERATIONS {
        for j in 0..NUM_GENERATIONS {
            if i == j { continue; }
            let mut p1 = generations[i].clone();
            let mut p2 = generations[j].clone();
            p1.lr = 0.0;
            p2.lr = 0.0;
            let mut score = 0i32;
            for _ in 0..EVAL_GAMES {
                score += play_game(&mut p1, &mut p2);
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
}
