mod tic_tac_toe;
use tic_tac_toe::*;

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
    let mut player1 = NeuralPlayer::new(0, 0.01);
    let mut player2 = NeuralPlayer::new(1, 0.01);

    let mut stats = [0i32; 3]; // [p1 wins, p2 wins, draws]

    for _ in 0..1000 {
        match play_game(&mut player1, &mut player2) {
            1 => stats[0] += 1,
            -1 => stats[1] += 1,
            0 => stats[2] += 1,
            _ => {}
        }
    }

    println!("P1 wins: {} P2 wins: {} Draws: {}", stats[0], stats[1], stats[2]);
}
