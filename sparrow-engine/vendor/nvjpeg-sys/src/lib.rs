//! Raw FFI Rust bindings to nvJPEG.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

mod bindings;

pub use bindings::*;

#[macro_export]
macro_rules! check {
    ($status:ident, $err:literal) => {
        if $status != 0 {
            Err(format!("{}. Error occured with code: {}", $err, $status))?
        }
    };
}
#[cfg(test)]
mod tests {

    use custos::{
        buf,
        cuda::launch_kernel,
        prelude::{CUBuffer, Float, Number},
        static_api::static_cuda,
        Buffer, CDatatype,
    };

    pub fn correlate_cu2<T: Number + CDatatype>(
        input: &CUBuffer<T>,
        filter: &CUBuffer<T>,
        out: &mut CUBuffer<T>,
        inp_rows: usize,
        inp_cols: usize,
        filter_rows: usize,
        filter_cols: usize,
    ) {
        let (out_rows, out_cols) = (inp_rows - filter_rows + 1, inp_cols - filter_cols + 1);

        const THREADS: u32 = 8;

        // THREADS
        let grid_x = (inp_rows as f32 / THREADS as f32).ceil() as u32;
        let grid_y = (inp_cols as f32 / THREADS as f32).ceil() as u32;
        //let grid_z = ( as f32 / THREADS as f32).ceil() as u32;

        let src = format!(
            r#"
        extern "C" __global__ void correlate2({dtype}* input, {dtype}* filter, {dtype}* out, int inp_rows, int inp_cols, int filter_rows, int filter_cols) {{
            int moveDown = blockDim.x * blockIdx.x + threadIdx.x;
            int moveRight = blockDim.y * blockIdx.y + threadIdx.y;

            int outRows = inp_rows - filter_rows + 1;
            int outCols = inp_cols - filter_cols + 1;

            if (moveDown >= outRows) {{
                return;
            }} 
            if (moveRight >= outCols) {{
                return;
            }}
            {dtype} sum = 0;
            for (int filterRow = 0; filterRow < filter_rows; filterRow++) {{
                int inputIdx = moveDown * inp_cols + moveRight + filterRow * inp_cols;  
                for (int filterCol = 0; filterCol < filter_cols; filterCol++) {{
                    sum += input[inputIdx + filterCol] * filter[filterRow * filter_cols + filterCol];
                }}
            }}
            out[moveDown * outCols + moveRight] = sum;
        }}
    "#,
            dtype = T::as_c_type_str()
        );

        launch_kernel(
            input.device(),
            [grid_x, grid_y, 1],
            [THREADS, THREADS, 1],
            0,
            &src,
            "correlate2",
            &[
                input,
                filter,
                out,
                &inp_rows,
                &inp_cols,
                &filter_rows,
                &filter_cols,
            ],
        )
        .unwrap();
    }

    pub fn correlate_cu_use_z<T: Number + CDatatype>(
        input: &CUBuffer<T>,
        filter: &CUBuffer<T>,
        out: &mut CUBuffer<T>,
        inp_rows: usize,
        inp_cols: usize,
        filter_rows: usize,
        filter_cols: usize,
    ) {
        let (out_rows, out_cols) = (inp_rows - filter_rows + 1, inp_cols - filter_cols + 1);

        const THREADS: u32 = 8;

        // THREADS
        let grid_x = (inp_rows as f32 / THREADS as f32).ceil() as u32;
        let grid_y = (inp_cols as f32 / THREADS as f32).ceil() as u32;
        //let grid_z = ( as f32 / THREADS as f32).ceil() as u32;

        let src = format!(
            r#"
        extern "C" __global__ void correlateWithZ({dtype}* input, {dtype}* filter, {dtype}* out, int inp_rows, int inp_cols, int filter_rows, int filter_cols) {{

            /*extern __shared__ {dtype} filterData[];

            for (int filterRow = 0; filterRow < filter_rows; filterRow++) {{
                for (int filterCol = 0; filterCol < filter_cols; filterCol++) {{
                    filterData[filterRow * filter_cols + filterCol] = filter[filterRow * filter_cols + filterCol];
                }}
            }}

            __syncthreads();*/



            int moveDown = blockDim.x * blockIdx.x + threadIdx.x;
            int moveRight = blockDim.y * blockIdx.y + threadIdx.y;
            //int filterRow = threadIdx.z;

            int outRows = inp_rows - filter_rows + 1;
            int outCols = inp_cols - filter_cols + 1;

            if (moveDown >= outRows) {{
                return;
            }} 
            if (moveRight >= outCols) {{
                return;
            }}
            {dtype} sum = 0;

            for (int filterRow = 0; filterRow < filter_rows; filterRow++) {{
                int inputIdx = moveDown * inp_cols + moveRight + filterRow * inp_cols;  
                for (int filterCol = 0; filterCol < filter_cols; filterCol++) {{
                    sum += input[inputIdx + filterCol] * filter[filterRow * filter_cols + filterCol];
                }}
            }}
            out[moveDown * outCols + moveRight] = sum;
        }}
    "#,
            dtype = T::as_c_type_str()
        );

        launch_kernel(
            input.device(),
            [grid_x, grid_y, 1],
            [THREADS, THREADS, 1],
            (filter_rows * filter_cols * std::mem::size_of::<T>()) as u32,
            &src,
            "correlateWithZ",
            &[
                input,
                filter,
                out,
                &inp_rows,
                &inp_cols,
                &filter_rows,
                &filter_cols,
            ],
        )
        .unwrap();
    }

    #[test]
    fn test_correleate_cu2_larger() {
        let height = 1080;
        let width = 1920;

        let data = (0..height * width)
            .into_iter()
            .map(|x| x as f32)
            .collect::<Vec<f32>>();
        let data = Buffer::from((static_cuda(), data));

        let filter_rows = 10;
        let filter_cols = 10;

        let filter = buf![1./3.; filter_rows * filter_cols].to_gpu();
        let mut out = buf![0.; (height-filter_rows+1) * (width-filter_cols+1)].to_gpu();

        correlate_cu2(
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
            &data.to_cpu(),
            (height, width),
            &filter.to_cpu(),
            (filter_rows, filter_cols),
            &mut cpu_out,
        );

        assert_eq_with_tolerance(&cpu_out.read(), &out.read(), 100.0);
    }

    #[test]
    fn test_correlate_cu_larger_assert() {
        #[rustfmt::skip]
    let height = 1080;
        let width = 1920;

        for height in 1080..=1080 {
            println!("height: {}", height);
            for width in 1920..=1920 {
                let data = (0..height * width)
                    .into_iter()
                    .map(|x| x as f32)
                    .collect::<Vec<f32>>();
                let data = Buffer::from((static_cuda(), data));

                let filter_rows = 10;
                let filter_cols = 10;

                let filter = buf![1./3.; filter_rows * filter_cols].to_gpu();
                let mut out = buf![0.; (height-filter_rows+1) * (width-filter_cols+1)].to_gpu();

                correlate_cu2(
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
                    &data.to_cpu(),
                    (height, width),
                    &filter.to_cpu(),
                    (filter_rows, filter_cols),
                    &mut cpu_out,
                );

                assert_eq_with_tolerance(&cpu_out.read(), &out.read(), 100.0);
            }
        }

        let data = (0..height * width)
            .into_iter()
            .map(|x| x as f32)
            .collect::<Vec<f32>>();
        let data = Buffer::from((static_cuda(), data));

        let filter = buf![1./3.; 9].to_gpu();
        let mut out = buf![0.; (height-3+1) * (width-3+1)].to_gpu();

        correlate_cu2(&data, &filter, &mut out, height, width, 3, 3);

        //println!("out: {out:?}");

        let mut cpu_out = buf![0.; out.len()];

        correlate_valid_mut(
            &data.to_cpu(),
            (height, width),
            &filter.to_cpu(),
            (3, 3),
            &mut cpu_out,
        );

        assert_eq_with_tolerance(&cpu_out.read(), &out.read(), 0.1);
    }

    pub fn assert_eq_with_tolerance<T: Float>(a: &[T], b: &[T], tolerance: T) {
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            if (a[i] - b[i]).abs() >= tolerance {
                panic!(
                    "
LHS SIDE: {:?}, 
            does not match with
RHS SIDE: {:?} which value?: {}, {}",
                    a, b, a[i], b[i]
                );
            }
        }
    }

    pub fn correlate_valid_mut<T: Number>(
        lhs_slice: &[T],
        lhs_dims: (usize, usize),
        kernel_slice: &[T],
        kernel_dims: (usize, usize),
        out: &mut [T],
    ) {
        let (lhs_rows, lhs_cols) = lhs_dims;
        let (kernel_rows, kernel_cols) = kernel_dims;

        let (out_rows, out_cols) = (lhs_rows - kernel_rows + 1, lhs_cols - kernel_cols + 1);

        //loop for row-axis (y)
        //moves multiplication 1 down
        for y in 0..out_rows {
            //loop for col-axis (x)
            //moves multiplication 1 to the right
            for x in 0..out_cols {
                let mut sum = T::default();
                //repeat kernel rows times to use move through all kernel rows
                for idx in 0..kernel_rows {
                    let index = idx * lhs_cols + x + y * lhs_cols;
                    let lhs_kernel_row = &lhs_slice[index..index + kernel_cols];

                    let index = idx * kernel_cols;
                    let kernel_row = &kernel_slice[index..index + kernel_cols];

                    for (i, value) in lhs_kernel_row.iter().enumerate() {
                        sum += *value * kernel_row[i];
                    }
                }
                // y * final_cols + x
                out[y * out_cols + x] = sum;
            }
        }
    }

    pub fn cu_padding<T: CDatatype>(
        input: &CUBuffer<T>,
        out: &mut CUBuffer<T>,
        inp_rows: usize,
        inp_cols: usize,
        x_padding: usize,
        y_padding: usize,
    ) {
        let grid_x = ((inp_cols + x_padding * 2) as f32 / 16.).ceil() as u32;
        let grid_y = ((inp_rows + y_padding * 2) as f32 / 16.).ceil() as u32;

        let src = format!(
            r#"
        extern "C" __global__ void addPadding({dtype}* input, {dtype}* out, int inpRows, int inpCols, int xPadding, int yPadding) {{
            int col = blockDim.x * blockIdx.x + threadIdx.x;
            int row = blockDim.y * blockIdx.y + threadIdx.y;

            if (row >= inpRows || col >= inpCols) {{
                return;
            }}

            out[yPadding * (inpCols + 2*xPadding) + row * (inpCols + 2 * xPadding) + col + xPadding] = input[row * inpCols + col];
        }}
    "#,
            dtype = T::as_c_type_str()
        );
        launch_kernel(
            input.device(),
            [grid_x, grid_y, 1],
            [16, 16, 1],
            0,
            &src,
            "addPadding",
            &[input, out, &inp_rows, &inp_cols, &x_padding, &y_padding],
        )
        .unwrap();
    }

    pub fn add_padding<T: Number>(
        inputs: &[T],
        inp_rows: usize,
        inp_cols: usize,
        x_padding: usize,
        y_padding: usize,
    ) -> Vec<T> {
        let mut padded_inputs =
            vec![T::zero(); (inp_rows + y_padding * 2) * (inp_cols + x_padding * 2)];

        for inp_row in 0..inp_rows {
            for inp_col in 0..inp_cols {
                padded_inputs[y_padding * (inp_cols + 2 * x_padding)
                    + x_padding
                    + inp_row * (inp_cols + 2 * x_padding)
                    + inp_col] = inputs[inp_row * inp_cols + inp_col];
            }
        }
        padded_inputs
    }

    #[test]
    fn test_cu_padding_to_cpu_padding() {
        let inp_rows = 1080;
        let inp_cols = 1920;
        let x_padding = 4;
        let y_padding = 4;

        let inputs = vec![1.; inp_rows * inp_cols];

        let padded_inputs = add_padding(&inputs, inp_rows, inp_cols, x_padding, y_padding);

        let mut gpu_inputs = buf![0.; inputs.len()].to_gpu();
        let mut gpu_padded_inputs = buf![0.; padded_inputs.len()].to_gpu();

        gpu_inputs.write(&inputs);
        cu_padding(
            &gpu_inputs,
            &mut gpu_padded_inputs,
            inp_rows,
            inp_cols,
            x_padding,
            y_padding,
        );

        /*for (idx, padded_val) in gpu_padded_inputs.read().iter().enumerate() {
            print!("{padded_val}, ");
            if (idx + 1) % (inp_cols + 2*x_padding) == 0 {
                println!()
            }
        }*/

        assert_eq_with_tolerance(&gpu_padded_inputs.read(), &padded_inputs, 0.1);
    }
}
