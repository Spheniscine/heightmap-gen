use std::{error::Error, fs::File};

use image::{codecs::png::PngEncoder, ImageEncoder};
use ndarray::Array2;
use pcg_mwc::Mwc256XXA64;
use rand::Rng;

trait CommonNumExt {
    fn div_ceil(self, b: Self) -> Self;
    fn div_floor(self, b: Self) -> Self;
    fn gcd(self, b: Self) -> Self;
    fn highest_one(self) -> Self;
    fn lowest_one(self) -> Self;
    fn sig_bits(self) -> u32;
}

macro_rules! impl_common_num_ext {
    ($($ix:tt = $ux:tt),*) => {
        $(
            impl CommonNumExt for $ux {
                fn div_ceil(self, b: Self) -> Self {
                    let q = self / b; let r = self % b;
                    if r != 0 { q + 1 } else { q }
                }
                fn div_floor(self, b: Self) -> Self { self / b }
                fn gcd(self, mut b: Self) -> Self {
                    let mut a = self;
                    if a == 0 || b == 0 { return a | b; }
                    let shift = (a | b).trailing_zeros();
                    a >>= a.trailing_zeros();
                    b >>= b.trailing_zeros();
                    while a != b {
                        if a > b { a -= b; a >>= a.trailing_zeros(); }
                        else { b -= a; b >>= b.trailing_zeros(); }
                    }
                    a << shift
                }
                #[inline] fn highest_one(self) -> Self { 
                    if self == 0 { 0 } else { const ONE: $ux = 1; ONE << self.sig_bits() - 1 } 
                }
                #[inline] fn lowest_one(self) -> Self { self & self.wrapping_neg() }
                #[inline] fn sig_bits(self) -> u32 { std::mem::size_of::<$ux>() as u32 * 8 - self.leading_zeros() }
            }

            impl CommonNumExt for $ix {
                fn div_ceil(self, b: Self) -> Self {
                    let q = self / b; let r = self % b;
                    if self ^ b >= 0 && r != 0 { q + 1 } else { q }
                }
                fn div_floor(self, b: Self) -> Self { 
                    let q = self / b; let r = self % b;
                    if self ^ b < 0 && r != 0 { q - 1 } else { q }
                }
                fn gcd(self, b: Self) -> Self {
                    fn w_abs(x: $ix) -> $ux { (if x.is_negative() { x.wrapping_neg() } else { x }) as _ }
                    w_abs(self).gcd(w_abs(b)) as _
                }
                #[inline] fn highest_one(self) -> Self { (self as $ux).highest_one() as _ }
                #[inline] fn lowest_one(self) -> Self { self & self.wrapping_neg() }
                #[inline] fn sig_bits(self) -> u32 { std::mem::size_of::<$ix>() as u32 * 8 - self.leading_zeros() }
            }
        )*
    }
}
impl_common_num_ext!(i8 = u8, i16 = u16, i32 = u32, i64 = u64, i128 = u128, isize = usize);

const HEIGHT: usize = 512;
const WIDTH: usize = 512;
const OCTAVES: usize = 8;
const ATTENUATION: f32 = 0.75;

type Point = (f32, f32);

fn dot_grid_gradient(grid: &Array2<Point>, ix: usize, iy: usize, x: f32, y: f32) -> f32 {
    let gradient = grid[[ix, iy]];

    let dx = x - ix as f32;
    let dy = y - iy as f32;

    dx * gradient.0 + dy * gradient.1
}

fn interpolate(a0: f32, a1: f32, w: f32) -> f32 {
    (a1 - a0) * (3. - w * 2.) * w * w + a0
}

fn perlin(src: &Array2<Point>, x: f32, y: f32) -> f32 {
    let x0 = x as usize;
    let y0 = y as usize;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let sx = x - x0 as f32;
    let sy = y - y0 as f32;

    let n0 = dot_grid_gradient(&src, x0, y0, x, y);
    let n1 = dot_grid_gradient(&src, x1, y0, x, y);
    let ix0 = interpolate(n0, n1, sx);

    let n0 = dot_grid_gradient(&src, x0, y1, x, y);
    let n1 = dot_grid_gradient(&src, x1, y1, x, y);
    let ix1 = interpolate(n0, n1, sx);

    interpolate(ix0, ix1, sy)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut rng = Mwc256XXA64::new(0x243F6A8885A308D3, 0x13198A2E03707344);

    let mut scale = 1f32;
    let mut scale_sum = 0f32;

    let mut res = Array2::from_elem([HEIGHT, WIDTH], 0f32);

    for level in (0..OCTAVES).rev() {
        let cell_size = 1 << level;

        let h = HEIGHT.div_ceil(cell_size);
        let w = WIDTH.div_ceil(cell_size);

        let mut vecs = Array2::from_elem([h+1, w+1], Point::default());
        for i in 0..=h { for j in 0..=w {
            vecs[[i, j]] = (rng.gen::<f32>() * std::f32::consts::TAU).sin_cos();
        }}

        for i in 0..HEIGHT { for j in 0..WIDTH {
            let x = i as f32 / cell_size as f32;
            let y = j as f32 / cell_size as f32;
            res[[i, j]] += perlin(&vecs, x, y) * scale;
        }}

        scale_sum += scale;
        scale *= ATTENUATION;
    }

    for i in 0..HEIGHT { for j in 0..WIDTH {
        res[[i, j]] /= scale_sum;
    }}

    let mut buf = vec![0u8; HEIGHT * WIDTH];
    for i in 0..HEIGHT { for j in 0..WIDTH {
        buf[i * WIDTH + j] = (res[[i, j]].clamp(-1., 1.) * 127.5 + 127.5).round() as u8;
    }}

    let mut writer = File::create("output.png")?;
    let encoder = PngEncoder::new(&mut writer);
    encoder.write_image(&buf, WIDTH as _, HEIGHT as _, image::ColorType::L8)?;

    Ok(())
}
