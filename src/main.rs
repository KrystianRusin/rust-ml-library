use rust_ml_library::linear_regression::LinearRegression;
use ndarray::{array, Array2, Array1};

fn main() {
    let x: Array2<f64> = array![
        [1.0, 1.0],  // Observation 1
        [1.0, 2.0],  // Observation 2
        [2.0, 2.0],  // Observation 3
        [2.0, 3.0]   // Observation 4
    ];
    let y: Array1<f64> = array![6.0, 8.0, 9.0, 11.0];

    let mut model = LinearRegression::new(0.01, 1000);
    model.fit(&x, &y);

    if let Some(predictions) = model.predict(&x) {
        println!("Predictions: {:?}", predictions);
    } else {
        println!("Model has not been trained yet.");
    }
}
