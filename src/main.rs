use burn::tensor::{backend::ndarray::NdArrayBackend, Tensor};

fn main() {
    type Backend = NdArrayBackend<f32>;

    // Create a simple tensor using the Burn machine learning crate
    let tensor = Tensor::<Backend, 1>::from_data([1.0, 2.0, 3.0]);
    println!("{:?}", tensor);
}
