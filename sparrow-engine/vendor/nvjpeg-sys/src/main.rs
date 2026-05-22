/*use std::time::Instant;

use custos::{buf, static_api::static_cuda, Buffer};
use nvjpeg_sys::{
    assert_eq_with_tolerance, correlate_cu2, correlate_cu_use_z, correlate_valid_mut,
};

fn main() {
    let height = 1080;
    let width = 1920;

    let data = (0..height * width)
        .into_iter()
        .map(|x| x as f32)
        .collect::<Vec<f32>>();
    let data = Buffer::from((static_cuda(), data));

    let filter_rows = 32;
    let filter_cols = 32;

    let filter = buf![1./3.; filter_rows * filter_cols].to_gpu();
    let mut out = buf![0.; (height-filter_rows+1) * (width-filter_cols+1)].to_gpu();

    correlate_cu_use_z(
        &data,
        &filter,
        &mut out,
        height,
        width,
        filter_rows,
        filter_cols,
    );

    //println!("out: {out:?}");

    let mut cpu_out = buf![0.; out.len()];

    correlate_valid_mut(
        &data.read(),
        (height, width),
        &filter.read(),
        (filter_rows, filter_cols),
        &mut cpu_out,
    );

    // assert_eq_with_tolerance(&cpu_out.read(), &out.read(), 100.0);

    println!("checked!");

    let start = Instant::now();
    for _ in 0..10 {
        correlate_cu_use_z(
            &data,
            &filter,
            &mut out,
            height,
            width,
            filter_rows,
            filter_cols,
        );
        static_cuda().stream().sync().unwrap();
    }

    println!(
        "Duration of correlate_cu2 (cached kernel): {:?}",
        start.elapsed()
    );
}
*/

fn main() {}
