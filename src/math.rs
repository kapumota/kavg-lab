/// Calcula el producto punto entre dos vectores.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Calcula ||x||².
pub fn norm2_squared(x: &[f64]) -> f64 {
    dot(x, x)
}

/// Calcula ||x||₂.
pub fn norm2(x: &[f64]) -> f64 {
    norm2_squared(x).sqrt()
}

/// Suma vectorial: a + b.
pub fn add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// Resta vectorial: a - b.
pub fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

/// Multiplicación escalar: c * x.
pub fn scale(c: f64, x: &[f64]) -> Vec<f64> {
    x.iter().map(|v| c * v).collect()
}

/// Suma a + c b sin crear operaciones intermedias innecesarias.
pub fn add_scaled(a: &[f64], c: f64, b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x + c * y).collect()
}

/// Multiplicación matriz-vector.
pub fn mat_vec(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    matrix.iter().map(|row| dot(row, x)).collect()
}

/// Multiplicación por la transpuesta: Aᵀx.
pub fn mat_t_vec(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    if matrix.is_empty() {
        return Vec::new();
    }

    let cols = matrix[0].len();
    let mut out = vec![0.0; cols];

    for (i, row) in matrix.iter().enumerate() {
        for j in 0..cols {
            out[j] += row[j] * x[i];
        }
    }

    out
}

/// Convierte un vector a una cadena compacta para CSV.
pub fn format_vector(x: &[f64]) -> String {
    let values: Vec<String> = x.iter().map(|v| format!("{:.10}", v)).collect();
    format!("[{}]", values.join(","))
}

/// Transpone una matriz densa representada como Vec<Vec<f64>>.
pub fn transpose(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if matrix.is_empty() {
        return Vec::new();
    }

    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut out = vec![vec![0.0; rows]; cols];

    for i in 0..rows {
        for j in 0..cols {
            out[j][i] = matrix[i][j];
        }
    }

    out
}

/// Resuelve Ax=b con eliminación gaussiana y pivoteo parcial.
/// Esta rutina es suficiente para los ejemplos pequeños .
pub fn solve_linear_system(matrix: &[Vec<f64>], rhs: &[f64]) -> anyhow::Result<Vec<f64>> {
    let n = matrix.len();
    anyhow::ensure!(n > 0, "La matriz no puede estar vacía.");
    anyhow::ensure!(
        matrix.iter().all(|row| row.len() == n),
        "La matriz debe ser cuadrada."
    );
    anyhow::ensure!(rhs.len() == n, "La dimensión de b no coincide con A.");

    let mut a = matrix.to_vec();
    let mut b = rhs.to_vec();

    for col in 0..n {
        let mut pivot = col;
        let mut pivot_abs = a[col][col].abs();

        let mut row = col + 1;
        while row < n {
            let candidate = a[row][col].abs();
            if candidate > pivot_abs {
                pivot = row;
                pivot_abs = candidate;
            }
            row += 1;
        }

        anyhow::ensure!(
            pivot_abs > 1.0e-14,
            "La matriz parece singular o mal condicionada."
        );

        if pivot != col {
            a.swap(pivot, col);
            b.swap(pivot, col);
        }

        let diag = a[col][col];
        let mut j = col;
        while j < n {
            a[col][j] /= diag;
            j += 1;
        }
        b[col] /= diag;

        for row in 0..n {
            if row == col {
                continue;
            }

            let factor = a[row][col];
            if factor.abs() <= 1.0e-18 {
                continue;
            }

            let mut j = col;
            while j < n {
                a[row][j] -= factor * a[col][j];
                j += 1;
            }
            b[row] -= factor * b[col];
        }
    }

    Ok(b)
}
