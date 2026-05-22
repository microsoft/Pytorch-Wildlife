use std::{io::Write, mem::size_of, ops::Mul, ptr::null_mut, time::Instant};

use custos::{
    buf,
    cuda::{api::culaunch_kernel, fn_cache, launch_kernel},
    prelude::{launch_kernel1d, CUBuffer, Float, Number},
    static_api::static_cuda,
    Buffer, CDatatype, Device, CUDA,
};
use nvjpeg_sys::{
    check, nvjpegChromaSubsampling_t, nvjpegCreateSimple, nvjpegDecode, nvjpegDestroy,
    nvjpegGetImageInfo, nvjpegHandle_t, nvjpegImage_t, nvjpegJpegStateCreate,
    nvjpegJpegStateDestroy, nvjpegJpegState_t, nvjpegOutputFormat_t_NVJPEG_OUTPUT_RGB,
};

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let raw_data = std::fs::read("cat_798x532.jpg")?;
    //let raw_data = std::fs::read("cat.jpg")?;

    let device = CUDA::new(0)?;

    let image = unsafe { decode_raw_jpeg(&raw_data, &device)? };

    Ok(())
}

#[derive(Debug, Default)]
pub struct Image {}

unsafe fn decode_raw_jpeg(
    raw_data: &[u8],
    device: &CUDA,
) -> Result<Image, Box<dyn std::error::Error + Send + Sync>> {
    let mut handle: nvjpegHandle_t = null_mut();

    let status = nvjpegCreateSimple(&mut handle);
    check!(status, "Could not create simple handle. ");

    let mut jpeg_state: nvjpegJpegState_t = null_mut();
    let status = nvjpegJpegStateCreate(handle, &mut jpeg_state);
    check!(status, "Could not create jpeg state. ");

    let mut n_components = 0;
    let mut subsampling: nvjpegChromaSubsampling_t = 0;
    let mut widths = [0, 0, 0];
    let mut heights = [0, 0, 0];

    let status = nvjpegGetImageInfo(
        handle,
        raw_data.as_ptr(),
        raw_data.len(),
        &mut n_components,
        &mut subsampling,
        widths.as_mut_ptr(),
        heights.as_mut_ptr(),
    );
    check!(status, "Could not get image info. ");

    heights[0] = heights[1] * 2;

    println!("n_components: {n_components}, subsampling: {subsampling}, widths: {widths:?}, heights: {heights:?}");

    let mut image: nvjpegImage_t = nvjpegImage_t::new();

    image.pitch[0] = widths[0] as usize;
    image.pitch[1] = widths[0] as usize;
    image.pitch[2] = widths[0] as usize;

    let channel0 = Buffer::<u8, _>::new(device, image.pitch[0] * heights[0] as usize);
    let channel1 = Buffer::<u8, _>::new(device, image.pitch[0] * heights[0] as usize);
    let channel2 = Buffer::<u8, _>::new(device, image.pitch[0] * heights[0] as usize);

    image.channel[0] = channel0.cu_ptr() as *mut _;
    image.channel[1] = channel1.cu_ptr() as *mut _;
    image.channel[2] = channel2.cu_ptr() as *mut _;

    let status = nvjpegDecode(
        handle,
        jpeg_state,
        raw_data.as_ptr(),
        raw_data.len(),
        nvjpegOutputFormat_t_NVJPEG_OUTPUT_RGB,
        &mut image,
        device.stream().0 as *mut _,
    );
    check!(status, "Could not decode image. ");

    //device.stream().sync()?;

    let channel0 = channel0.read();
    let channel1 = channel1.read();
    let channel2 = channel2.read();

    let filter_rows = 300;
    let filter_cols = 300;
    let filter = vec![1. / (filter_rows * filter_cols) as f32; filter_rows * filter_cols];

    let mut channel0_out = vec![0.; channel0.len()];
    correlate_fully(
        &channel0.iter().map(|x| *x as f32).collect::<Vec<_>>(),
        &filter,
        &mut channel0_out,
        heights[0] as usize,
        widths[0] as usize,
        filter_rows,
        filter_cols,
    );

    let mut channel1_out = vec![0.; channel1.len()];
    correlate_fully(
        &channel1.iter().map(|x| *x as f32).collect::<Vec<_>>(),
        &filter,
        &mut channel1_out,
        heights[0] as usize,
        widths[0] as usize,
        filter_rows,
        filter_cols,
    );

    let mut channel2_out = vec![0.; channel2.len()];
    correlate_fully(
        &channel2.iter().map(|x| *x as f32).collect::<Vec<_>>(),
        &filter,
        &mut channel2_out,
        heights[0] as usize,
        widths[0] as usize,
        filter_rows,
        filter_cols,
    );

    let channel0_out = channel0_out
        .into_iter()
        .map(|x| x as u8)
        .collect::<Vec<_>>();
    let channel1_out = channel1_out
        .into_iter()
        .map(|x| x as u8)
        .collect::<Vec<_>>();
    let channel2_out = channel2_out
        .into_iter()
        .map(|x| x as u8)
        .collect::<Vec<_>>();

    let file = std::fs::File::create("cat_798x532.ppm")?;
    let mut writer = std::io::BufWriter::new(file);
    writer.write(format!("P6\n{} {}\n255\n", widths[0], heights[0]).as_bytes())?;

    for row in 0..heights[0] {
        let row = row as usize;
        for col in 0..widths[0] {
            let col = col as usize;
            writer.write(&[
                channel0_out[row * widths[0] as usize + col],
                channel1_out[row * widths[0] as usize + col],
                channel2_out[row * widths[0] as usize + col],
            ])?;
        }
    }

    writer.flush()?;

    // free
    /*
    let status = nvjpegJpegStateDestroy(jpeg_state);
    check!(status, "Could not free jpeg state. ");

    let status = nvjpegDestroy(handle);
    check!(status, "Could not free nvjpeg handle. ");*/

    Ok(Image::default())
}

pub fn correlate_cu_padded<T: Number + CDatatype>(
    input: &CUBuffer<T>,
    filter: &CUBuffer<T>,
    out: &mut CUBuffer<T>,
    inp_rows: usize,
    inp_cols: usize,
    filter_rows: usize,
    filter_cols: usize,
) {
    let x_padding = filter_cols - 1;
    let y_padding = filter_rows - 1;

    let inp_rows2 = inp_rows + y_padding * 2;
    let inp_cols2 = inp_cols + x_padding * 2;

    let src = format!(
        r#"
        extern "C" __global__ void correlate({dtype}* input, {dtype}* filter, {dtype}* out, int inp_rows, int inp_cols, int unpad_rows, int unpad_cols, int filter_rows, int filter_cols) {{
            int idx = blockDim.x * blockIdx.x + threadIdx.x;

            int xPadding = filter_cols - 1;
            int yPadding = filter_rows - 1;

            int moveDown = blockIdx.x / (inp_cols-filter_cols + 1);
            int moveRight = blockIdx.x % (inp_cols-filter_cols + 1);
            int startOfBlock = moveDown * inp_cols + moveRight;
            int next = startOfBlock + (threadIdx.x / filter_cols) * inp_cols + (threadIdx.x % filter_cols);

            (yPadding + unpad_rows) * inp_cols

            if (next < 0) {{
                return;
            }}

            //(yPadding + ) * inp_cols 

            if (next >= inp_rows * inp_cols || next < 0) {{
              //  return;
            }}

            //__shared__ {dtype} res[filter_rows * filter_cols];
            extern __shared__ {dtype} res[];

            res[threadIdx.x] = input[next] * filter[threadIdx.x];

            __syncthreads();

            // sum res and write to out
            if (threadIdx.x == 0) {{
                {dtype} sum = 0;
                for (int i = 0; i < filter_rows * filter_cols; i++) {{
                    sum += res[i];
                }}
                out[blockIdx.x] = sum;
            }} 
            //printf("block: %d, idx: %d, val: %f  \n", blockIdx.x, next, input[next]);
            //printf("shared val: %f  \n", res[0]);
        }} 
    "#,
        dtype = T::as_c_type_str()
    );

    launch_kernel(
        input.device(),
        [
            ((inp_rows2 - filter_rows + 1) * (inp_cols2 - filter_cols + 1)) as u32,
            1,
            1,
        ],
        [(filter_cols * filter_rows) as u32, 1, 1],
        (filter_cols * filter_rows * size_of::<T>()) as u32,
        &src,
        "correlate",
        &[
            input,
            filter,
            out,
            &inp_rows2,
            &inp_cols2,
            &inp_rows,
            &inp_cols,
            &filter_rows,
            &filter_cols,
        ],
    )
    .unwrap();
}

#[test]
fn test_correlate_cu_padded() {
    #[rustfmt::skip]
    let data = buf![
        1., 2., 3., 4., 
        5., 6., 7., 8., 
        9., 10., 11., 12., 
        13., 14., 15., 16., 
        17., 18., 19., 20.
    ]
    .to_gpu();

    let filter = buf![1.; 9].to_gpu();
    let mut out = buf![0.; data.len()].to_gpu();

    correlate_cu_padded(&data, &filter, &mut out, 5, 4, 3, 3);
    println!("out: {out:?}");
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
pub fn correlate_cu<T: Number + CDatatype>(
    input: &CUBuffer<T>,
    filter: &CUBuffer<T>,
    out: &mut CUBuffer<T>,
    inp_rows: usize,
    inp_cols: usize,
    filter_rows: usize,
    filter_cols: usize,
) {
    let src = format!(
        r#"
        extern "C" __global__ void correlate({dtype}* input, {dtype}* filter, {dtype}* out, int inp_rows, int inp_cols, int filter_rows, int filter_cols) {{
            int idx = blockDim.x * blockIdx.x + threadIdx.x;

            int moveDown = blockIdx.x / (inp_cols-filter_cols + 1);
            //int moveRight = blockIdx.x % (inp_cols - filter_cols + 1);
            int moveRight = blockIdx.x % (inp_cols-filter_cols + 1);
            int startOfBlock = moveDown * inp_cols + moveRight;
            int next = startOfBlock + (threadIdx.x / filter_cols) * inp_cols + (threadIdx.x % filter_cols);

            //__shared__ {dtype} res[filter_rows * filter_cols];
            extern __shared__ {dtype} res[];

            res[threadIdx.x] = input[next] * filter[threadIdx.x];

            __syncthreads();

            // sum res and write to out
            if (threadIdx.x == 0) {{
                {dtype} sum = 0;
                for (int i = 0; i < filter_rows * filter_cols; i++) {{
                    sum += res[i];
                }}
                out[blockIdx.x] = sum;
            }}
            //printf("block: %d, idx: %d, val: %f  \n", blockIdx.x, next, input[next]);
            //printf("shared val: %f  \n", res[0]);
        }} 
    "#,
        dtype = T::as_c_type_str()
    );

    launch_kernel(
        input.device(),
        [
            ((inp_rows - filter_rows + 1) * (inp_cols - filter_cols + 1)) as u32,
            1,
            1,
        ],
        [(filter_cols * filter_rows) as u32, 1, 1],
        (filter_cols * filter_rows * size_of::<T>()) as u32,
        &src,
        "correlate",
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
fn test_correlate_cu() {
    let data = buf![
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20.
    ]
    .to_gpu();

    let filter = buf![1.; 9].to_gpu();

    let mut out = buf![0.; data.len()].to_gpu();

    correlate_cu(&data, &filter, &mut out, 5, 4, 3, 3);
    println!("out: {out:?}");
}

#[test]
fn test_correlate_cu_larger() {
    let rows = 1920;
    let cols = 1080;

    let data = buf![
        1.4; rows * cols
    ]
    .to_gpu();

    let filter_rows = 32;
    let filter_cols = filter_rows;

    let filter = buf![1.; filter_rows * filter_cols].to_gpu();

    let mut out = buf![0.; (rows - filter_rows + 1) * (cols - filter_cols +1)].to_gpu();
    correlate_cu2(
        &data,
        &filter,
        &mut out,
        rows,
        cols,
        filter_rows,
        filter_cols,
    );
    out.device().stream().sync().unwrap();
    let start = std::time::Instant::now();
    correlate_cu2(
        &data,
        &filter,
        &mut out,
        rows,
        cols,
        filter_rows,
        filter_cols,
    );
    out.device().stream().sync().unwrap();
    println!("elapsed: {:?}", start.elapsed());
    //println!("out: {out:?}");
}

#[test]
fn test_cu_padding_la() {
    let data = buf![1; 10000*14000].to_cuda();

    let mut out = buf![0; (10000 + 2*2) * (14000+ 2*2)].to_cuda();
    cu_padding(&data, &mut out, 10000, 14000, 2, 2);
    data.device().stream().sync().unwrap();

    let start = Instant::now();
    cu_padding(&data, &mut out, 10000, 14000, 2, 2);
    data.device().stream().sync().unwrap();
    println!("elapsed: {:?}", start.elapsed());
}

#[test]
fn test_cu_padding() {
    #[rustfmt::skip]
    let data = buf![
        1, 2, 3, 5, 
        4, 3, 2, 1, 
        8, 7, 4, 2, 
        7, 3, 2, 1, 
        8, 5, 3, 8
    ].to_cuda();

    let mut out = buf![0; (5 + 2*2) * (4+ 2*2)].to_cuda();
    cu_padding(&data, &mut out, 5, 4, 2, 2);
    println!("out: {out:?}");

    for (idx, padded_val) in out.to_cpu().iter().enumerate() {
        print!("{padded_val}, ");
        if (idx + 1) % (2 + 2 + 4) == 0 {
            println!()
        }
    }
}

#[test]
fn test_add_padding() {
    #[rustfmt::skip]
    let data = [
        1, 2, 3, 5, 
        4, 3, 2, 1, 
        8, 7, 4, 2, 
        7, 3, 2, 1, 
        8, 5, 3, 8
    ];

    let padded = add_padding(&data, 5, 4, 2, 2);

    for (idx, padded_val) in padded.iter().enumerate() {
        print!("{padded_val}, ");
        if (idx + 1) % (2 + 2 + 4) == 0 {
            println!()
        }
    }
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

    const THREADS: u32 = 32;

    // THREADS
    let grid_x = (inp_rows as f32 / THREADS as f32).ceil() as u32;
    let grid_y = (inp_cols as f32 / THREADS as f32).ceil() as u32;

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

#[test]
fn test_correlate_cu_2() {
    #[rustfmt::skip]
    let data = buf![
        1., 2., 3., 4., 
        5., 6., 7., 8., 
        9., 10., 11., 12., 
        13., 14., 15., 16., 
        17., 18., 19., 20.
    ]
    .to_gpu();

    let filter = buf![1./3.; 9].to_gpu();
    let mut out = buf![0.; (5-3+1) * (4-3+1)].to_gpu();

    correlate_cu2(&data, &filter, &mut out, 5, 4, 3, 3);

    println!("out: {out:?}");

    let mut cpu_out = buf![0.; out.len()];

    correlate_valid_mut(
        &data.to_cpu(),
        (5, 4),
        &filter.to_cpu(),
        (3, 3),
        &mut cpu_out,
    );
    assert_eq!(cpu_out.read(), out.read());
}

#[test]
fn test_correlate_cu_larger_assert() {
    #[rustfmt::skip]
    let height = 1080;
    let width = 1920;

    for height in 950..1100 {
        for width in 1500..1980 {
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

pub fn correlate_cu2_pad<T: Number + CDatatype>(
    input: &CUBuffer<T>,
    filter: &CUBuffer<T>,
    out: &mut CUBuffer<T>,
    inp_rows: usize,
    inp_cols: usize,
    filter_rows: usize,
    filter_cols: usize,
) {
    const THREADS: u32 = 8;

    let x_padding = filter_cols - 1;
    let y_padding = filter_rows - 1;

    let inp_rows2 = inp_rows + y_padding * 2;
    let inp_cols2 = inp_cols + x_padding * 2;

    let (out_rows, out_cols) = (inp_rows2 - filter_rows + 1, inp_cols2 - filter_cols + 1);

    let grid_x = (out_rows as f32 / THREADS as f32).ceil() as u32;
    let grid_y = (out_cols as f32 / THREADS as f32).ceil() as u32;

    let src = format!(
        r#"
        extern "C" __global__ void correlate2({dtype}* input, {dtype}* filter, {dtype}* out, int inp_rows, int inp_cols, int unpad_rows, int unpad_cols, int filter_rows, int filter_cols) {{
            int moveDown = blockDim.x * blockIdx.x + threadIdx.x;
            int moveRight = blockDim.y * blockIdx.y + threadIdx.y;

            int xPadding = filter_cols - 1;
            int yPadding = filter_rows - 1;

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
                int inputIdx = moveDown * inp_cols 
                    + moveRight + filterRow * inp_cols
                    - yPadding * inp_cols - xPadding;
                    // - (outCols % moveRight == 0) * xPadding;

                    // + inp_row * (inp_cols + 2 * x_padding)
                    // + inp_col

                

                for (int filterCol = 0; filterCol < filter_cols; filterCol++) {{

                    if (inputIdx +filterCol >= unpad_rows * unpad_cols) {{
                        continue;
                    }}
                    if (inputIdx+filterCol < 0) {{
                        continue;
                    }}

                    // print the input index + filterCol and the corresponding input value:
                    printf("%d, %f\n", inputIdx+filterCol, input[inputIdx+filterCol]);
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
            &inp_rows2,
            &inp_cols2,
            &inp_rows,
            &inp_cols,
            &filter_rows,
            &filter_cols,
        ],
    )
    .unwrap();
}

#[test]
fn test_correlate_cu_2_pad() {
    #[rustfmt::skip]
    let data = buf![
        1., 2., 9., 
        4., 5., 6., 
        7., 8., 9., 
    ]
    .to_gpu();

    let filter = buf![1.; 4].to_gpu();
    let mut out = buf![0.; data.len()].to_gpu();

    correlate_cu2_pad(&data, &filter, &mut out, 3, 3, 2, 2);

    println!("out: {out:?}");

    let mut cpu_out = buf![0.; out.len()];

    correlate_fully(&data.to_cpu(), &filter.to_cpu(), &mut cpu_out, 3, 3, 2, 2);
    println!("cpu out: {cpu_out:?}");
    // assert_eq!(cpu_out.read(), out.read());
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

pub fn correlate_fully<T: Number + Mul<U, Output = T>, U: Number>(
    inputs: &[T],
    filter: &[U],
    out: &mut [T],
    inp_rows: usize,
    inp_cols: usize,
    filter_rows: usize,
    filter_cols: usize,
) {
    let x_padding = filter_cols - 1;
    let y_padding = filter_rows - 1;

    let padded_inputs = add_padding(inputs, inp_rows, inp_cols, x_padding, y_padding);
    let padded_rows = inp_rows + y_padding * 2;
    let padded_cols = inp_cols + x_padding * 2;

    // attention: leaves the last padded row, col out
    for move_down in 0..=padded_rows - filter_rows - y_padding {
        for move_right in 0..=padded_cols - filter_cols - x_padding {
            let mut sum = T::default();
            for idx in 0..filter_rows {
                let filter_idx = idx * filter_cols;
                let filter_row = &filter[filter_idx..filter_idx + filter_cols];

                let input_idx = move_down * padded_cols + move_right + idx * padded_cols;
                let input_row = &padded_inputs[input_idx..input_idx + filter_cols];

                for (filter_row, input_row) in input_row.iter().zip(filter_row) {
                    sum += *filter_row * *input_row;
                }
            }
            out[move_down * inp_cols + move_right] = sum;
        }
    }
}

#[test]
fn test_correlate() {
    let data = [
        1., 2., 3., 5., 4., 3., 2., 1., 8., 7., 4., 2., 7., 3., 2., 1., 8., 5., 3., 8.,
    ];

    let filter = [1. / 4., 1. / 4., 1. / 4., 1. / 4.];
    let mut out = [0.; 5 * 4];
    correlate_fully(&data, &filter, &mut out, 5, 4, 2, 2);

    println!("out: {out:?}");

    let filter = [1. / 4.; 9];
    let mut out = [0.; 5 * 4];
    correlate_fully(&data, &filter, &mut out, 5, 4, 3, 3);

    println!("out: {out:?}");
}

#[test]
fn test_correlate_larger() {
    let rows = 5080;
    let cols = 520;

    let data = buf![
        1.4; rows * cols
    ];

    let filter_rows = 32;
    let filter_cols = filter_rows;

    let filter = buf![1.; filter_rows * filter_cols];

    let mut out = buf![0.; rows * cols];

    correlate_fully(
        &data,
        &filter,
        &mut out,
        rows,
        cols,
        filter_rows,
        filter_cols,
    );
    println!("out: {out:?}");
}
