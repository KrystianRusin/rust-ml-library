use ndarray::{Array1, Array2};

pub struct LinearRegression {
    coefficients: Option<Array1<f64>>,
    learning_rate: f64,
    iterations: usize,
}

impl LinearRegression {
    pub fn new(learning_rate: f64, iterations: usize) -> Self {
        LinearRegression { 
            coefficients: None,
            learning_rate,
            iterations,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        println!("{:?}", x.dim());
        let (m, n) = x.dim();
        let mut beta = Array1::<f64>::zeros(n);

        for _ in 0..self.iterations {
            let y_pred = x.dot(&beta);
            let error = y - &y_pred;
            let gradient = -2.0/(m as f64) * x.t().dot(&error);
            beta = beta - self.learning_rate * gradient;
        }

        self.coefficients = Some(beta)
    }

    pub fn predict(&self, x: &Array2<f64>) -> Option<Array1<f64>> {
        self.coefficients.as_ref().map(|coeffs| x.dot(coeffs))
    }
}