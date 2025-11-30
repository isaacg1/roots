use image::{ImageBuffer, RgbImage};
use rand::prelude::*;
use std::collections::HashSet;

const EPSILON: f64 = 1e-8;

#[derive(Debug)]
struct Poly {
    // LSB first
    coeffs: Vec<i64>,
}
#[derive(Debug, Clone, Copy)]
struct Complex {
    r: f64,
    i: f64,
}
const ZERO: Complex = Complex { r: 0.0, i: 0.0 };
const ONE: Complex = Complex { r: 1.0, i: 0.0 };
impl Complex {
    fn from_int(int: i64) -> Self {
        Self {
            r: int as f64,
            i: 0.0,
        }
    }
    fn add(&self, other: Self) -> Self {
        Self {
            r: self.r + other.r,
            i: self.i + other.i,
        }
    }
    fn sub(&self, other: Self) -> Self {
        Self {
            r: self.r - other.r,
            i: self.i - other.i,
        }
    }
    fn mul(&self, other: Self) -> Self {
        Self {
            r: self.r * other.r - self.i * other.i,
            i: self.r * other.i + self.i * other.r,
        }
    }
    fn div(&self, other: Self) -> Option<Self> {
        let divisor = other.r.powi(2) + other.i.powi(2);
        if divisor == 0.0 {
            return None;
        }
        Some(Self {
            r: (self.r * other.r + self.i * other.i) / divisor,
            i: (self.i * other.r - self.r * other.i) / divisor,
        })
    }
}
impl Poly {
    fn apply(&self, start: Complex) -> Complex {
        let mut running = ZERO;
        let mut power = ONE;
        for &coeff in &self.coeffs {
            running = running.add(power.mul(Complex::from_int(coeff)));
            power = power.mul(start);
        }
        running
    }
    fn apply_deriv(&self, start: Complex) -> Complex {
        let mut running = ZERO;
        let mut power = ONE;
        for (place, &coeff) in self.coeffs.iter().enumerate().skip(1) {
            running = running.add(Complex::from_int(place as i64 * coeff).mul(power));
            power = power.mul(start);
        }
        running
    }
    fn newton(&self, start: Complex) -> Option<Complex> {
        Some(start.sub(self.apply(start).div(self.apply_deriv(start))?))
    }
    fn root(&self, start: Complex, max_iters: u64) -> Option<Complex> {
        let mut current = start;
        for _ in 0..max_iters {
            let old = current;
            current = self.newton(current)?;
            if (current.r - old.r).abs() < EPSILON && (current.i - old.i).abs() < EPSILON {
                return Some(current);
            }
            if current.r.is_nan()
                || current.i.is_nan()
                || current.r.abs() > 1e8
                || current.i.abs() > 1e8
            {
                return None;
            }
        }
        None
    }
}

fn choose(n: u64, k: u64) -> u64 {
    let mut running = 1;
    for i in 0..k {
        running *= n - i;
        running /= i + 1;
    }
    running
}

// x^17 + 2x - 5:
// degree 17, sum 8, nonzero 3
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
struct Params {
    degree: u64,
    sum: u64,
    nonzero: u64,
}
impl Params {
    fn new(degree: u64, sum: u64, nonzero: u64) -> Self {
        assert!(nonzero >= 1);
        assert!(nonzero <= degree + 1);
        assert!(nonzero <= sum);
        Self {
            degree,
            sum,
            nonzero,
        }
    }
    // d2, s3, n2
    // 2x^2+x, x^2+2x, 2x^2+1, x^2+2, *4
    fn count(&self) -> u64 {
        let nonzero_positions = choose(self.degree, self.nonzero - 1);
        let pos_coeffs = choose(self.sum - 1, self.nonzero - 1);
        nonzero_positions * pos_coeffs << self.nonzero
    }
    fn to_iter(&self) -> ParamIterator {
        ParamIterator::new(*self)
    }
}

struct ParamIterator {
    params: Params,
    positions: Vec<usize>,
    coeffs: Vec<u64>,
    signs: u64,
    cont: bool,
}
impl ParamIterator {
    fn new(params: Params) -> Self {
        let mut positions: Vec<usize> = (0..params.nonzero as usize - 1).collect();
        positions.push(params.degree as usize);
        let mut coeffs = vec![1; params.nonzero as usize];
        coeffs[0] += params.sum - params.nonzero;
        Self {
            params,
            positions,
            coeffs,
            signs: 0,
            cont: true,
        }
    }
}
impl Iterator for ParamIterator {
    type Item = Poly;
    fn next(&mut self) -> Option<Self::Item> {
        if !self.cont {
            return None;
        }
        let mut poly = vec![0; self.params.degree as usize + 1];
        for (i, &position) in self.positions.iter().enumerate() {
            let sign = (self.signs & (1 << i)) == 0;
            let coeff = if sign {
                self.coeffs[i] as i64
            } else {
                -(self.coeffs[i] as i64)
            };
            poly[position] = coeff
        }
        if self.signs < (1 << self.params.nonzero) - 1 {
            self.signs += 1
        } else {
            self.signs = 0;
            let first_nonone = self.coeffs.iter().position(|&c| c > 1);
            if let Some(nonone) = first_nonone
                && nonone < self.coeffs.len() - 1
            {
                let moving = self.coeffs[nonone] - 1;
                self.coeffs[nonone] = 1;
                self.coeffs[0] += moving - 1;
                self.coeffs[nonone + 1] += 1;
            } else {
                self.coeffs = vec![1; self.params.nonzero as usize];
                self.coeffs[0] += self.params.sum - self.params.nonzero;
                let first_gap = self.positions.windows(2).position(|w| w[0] < w[1] - 1);
                if let Some(gap) = first_gap {
                    self.positions[gap] += 1;
                    if gap > 0 {
                        for pos in 0..gap - 1 {
                            self.positions[pos] = pos
                        }
                    }
                } else {
                    self.cont = false;
                }
            }
        }
        Some(Poly { coeffs: poly })
    }
}
struct Histo {
    counts: Vec<u64>,
}
impl Histo {
    fn new() -> Self {
        Self { counts: vec![] }
    }
    fn add(&mut self, i: u64) {
        while self.counts.len() <= i as usize {
            self.counts.push(0)
        }
        self.counts[i as usize] += 1
    }
    fn hist(&self, buckets: usize) -> Vec<u64> {
        let total: u64 = self.counts.iter().sum();
        let mut out = vec![];
        let mut running = 0;
        for (i, count) in self.counts.iter().enumerate() {
            running += count;
            while running as usize * buckets / total as usize > out.len() {
                out.push(i as u64);
            }
        }
        out
    }
}

fn draw(
    pixels: usize,
    comp_size: f64,
    trials: u64,
    move_on: u64,
    max_iters: u64,
    seed: u64,
) -> RgbImage {
    let mut grid: Vec<Vec<Option<Params>>> = vec![vec![None; pixels]; pixels];
    let mut rng = StdRng::seed_from_u64(seed);
    let mut params_frontier: HashSet<Params> = HashSet::new();
    // Nonzero 1 gets stuck.
    params_frontier.insert(Params::new(1, 2, 2));
    let mut current_trials = 0;
    //let mut highest_degree = 0;
    //let mut highest_sum = 0;
    //let mut highest_nonzero = 0;
    let mut counts_degree = Histo::new();
    let mut counts_sum = Histo::new();
    let mut counts_nonzero = Histo::new();
    while current_trials < trials {
        let current_params = *params_frontier
            .iter()
            .min_by_key(|p| p.count())
            .expect("occupied");
        params_frontier.remove(&current_params);
        // Add new frontiers
        params_frontier.insert(Params::new(
            current_params.degree + 1,
            current_params.sum,
            current_params.nonzero,
        ));
        params_frontier.insert(Params::new(
            current_params.degree,
            current_params.sum + 1,
            current_params.nonzero,
        ));
        if current_params.nonzero < current_params.degree + 1
            && current_params.nonzero < current_params.sum
        {
            params_frontier.insert(Params::new(
                current_params.degree,
                current_params.sum,
                current_params.nonzero + 1,
            ));
        }
        // generate polys
        for poly in current_params.to_iter() {
            // search polys
            // Accelerate by skipping polys with no constant
            // Accelerate by skipping polys with negative constant term, adding negation
            if poly.coeffs[0] <= 0 {
                continue;
            }
            let mut fails = 0;
            while fails < move_on {
                current_trials += 1;
                fails += 1;
                let start = Complex {
                    r: rng.random_range(-comp_size..comp_size),
                    i: rng.random_range(-comp_size..comp_size),
                };
                let maybe_root = poly.root(start, max_iters);
                if let Some(root) = maybe_root {
                    let r_coord =
                        (root.r + comp_size) * (pixels as f64 / (2.0 * comp_size)).round();
                    let i_coord =
                        (root.i + comp_size) * (pixels as f64 / (2.0 * comp_size)).round();
                    if r_coord >= 0.0
                        && r_coord < pixels as f64
                        && i_coord >= 0.0
                        && i_coord < pixels as f64
                    {
                        let rc_int = r_coord as usize;
                        let ic_int = i_coord as usize;
                        if grid[rc_int][ic_int].is_none() {
                            grid[rc_int][ic_int] = Some(current_params);
                            grid[pixels-1-rc_int][pixels-1-ic_int] = Some(current_params);
                            fails = 0;
                            counts_degree.add(current_params.degree);
                            counts_sum.add(current_params.sum);
                            counts_nonzero.add(current_params.nonzero);
                            /*
                            highest_degree = highest_degree.max(current_params.degree);
                            highest_sum = highest_sum.max(current_params.sum);
                            highest_nonzero = highest_nonzero.max(current_params.nonzero);
                            */
                        }
                    }
                }
            }
        }
    }
    let hist_degree = counts_degree.hist(255);
    let hist_sum = counts_sum.hist(255);
    let hist_nonzero = counts_nonzero.hist(255);
    // Make a histogram of parameter frequencies, scale colors proportional to position in the histogram.
    let mut img: RgbImage = ImageBuffer::new(pixels as u32, pixels as u32);
    for (i, row) in grid.iter().enumerate() {
        for (j, maybe_params) in row.iter().enumerate() {
            if let Some(params) = maybe_params {
                let degree_color = hist_degree
                    .iter()
                    .filter(|&&deg| deg < params.degree)
                    .count() as u8;
                let sum_color = hist_sum.iter().filter(|&&deg| deg < params.sum).count() as u8;
                let nonzero_color = hist_nonzero
                    .iter()
                    .filter(|&&deg| deg < params.nonzero)
                    .count() as u8;
                let color = [degree_color, sum_color, nonzero_color];
                /*[
                    (params.degree * 255 / highest_degree) as u8,
                    (params.sum * 255 / highest_sum) as u8,
                    (params.nonzero * 255 / highest_nonzero) as u8,
                ];*/
                img.put_pixel(i as u32, j as u32, image::Rgb(color))
            }
        }
    }
    img
}

// Parameter set: sum, sparsity, degree.
// Generate frontier, consisting of all possible ways to increase any explored set by 1. Calculate size of all frontier sets. Add smallest, repeatedly.

fn main() {
    /*
    let params = Params::new(2, 3, 2);
    println!("count {}", params.count());
    for (i, poly) in params.to_iter().enumerate() {
        println!("{i} {poly:?}");
    }
    */
    let pixels = std::env::args()
        .nth(1)
        .expect("present")
        .parse()
        .expect("num");
    let comp_size = std::env::args()
        .nth(2)
        .expect("present")
        .parse()
        .expect("num");
    let trials = std::env::args()
        .nth(3)
        .expect("present")
        .parse()
        .expect("num");
    let move_on = std::env::args()
        .nth(4)
        .expect("present")
        .parse()
        .expect("num");
    let max_iters = std::env::args()
        .nth(5)
        .expect("present")
        .parse()
        .expect("num");
    let seed = std::env::args()
        .nth(6)
        .expect("present")
        .parse()
        .expect("num");
    let filename = format!("img-{pixels}-{comp_size}-{trials}-{move_on}-{max_iters}-{seed}.png");
    let image = draw(pixels, comp_size, trials, move_on, max_iters, seed);
    image.save(&filename).expect("saved");
}
