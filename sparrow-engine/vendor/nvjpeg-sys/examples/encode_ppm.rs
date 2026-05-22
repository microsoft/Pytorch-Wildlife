use std::ptr::null_mut;

use custos::{Buffer, CUDA};
use nvjpeg_sys::{
    check, nvjpegChromaSubsampling_t_NVJPEG_CSS_410, nvjpegChromaSubsampling_t_NVJPEG_CSS_420,
    nvjpegChromaSubsampling_t_NVJPEG_CSS_444, nvjpegCreateSimple, nvjpegEncodeImage,
    nvjpegEncodeRetrieveBitstream, nvjpegEncoderParamsCreate,
    nvjpegEncoderParamsSetSamplingFactors, nvjpegEncoderParams_t, nvjpegEncoderStateCreate,
    nvjpegEncoderState_t, nvjpegHandle_t, nvjpegImage_t, nvjpegInputFormat_t_NVJPEG_INPUT_RGB,
    nvjpegOutputFormat_t_NVJPEG_OUTPUT_RGB,
};

pub struct PPMImage {
    width: i32,
    height: i32,
    r: Vec<u8>,
    g: Vec<u8>,
    b: Vec<u8>,
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let raw_data = std::fs::read("cat_798x532.ppm")?;

    let mut data = Vec::with_capacity(16);
    let mut stop = 0;
    for raw_datum in &raw_data {
        if *raw_datum == '\n' as u8 {
            if stop == 1 {
                break;
            }
            stop += 1;
        }
        data.push(*raw_datum);
    }

    let (_, img_desc_str) = std::str::from_utf8(&data)
        .unwrap()
        .split_once('\n')
        .unwrap();
    let (width, height) = img_desc_str.split_once(' ').unwrap();
    let width = width.parse::<i32>().unwrap();
    let height = height.parse::<i32>().unwrap();

    debug_assert_eq!(width as usize, 798);
    debug_assert_eq!(height as usize, 532);

    let mut r = Vec::with_capacity((width * height) as usize);
    let mut g = Vec::with_capacity((width * height) as usize);
    let mut b = Vec::with_capacity((width * height) as usize);

    for pixel in raw_data[data.len() + 5..].chunks(3) {
        debug_assert_eq!(pixel.len(), 3);
        r.push(pixel[0]);
        g.push(pixel[1]);
        b.push(pixel[2]);
    }

    debug_assert_eq!(r.len(), (width * height) as usize);

    let ppm_img = PPMImage {
        width,
        height,
        r,
        g,
        b,
    };

    //println!("w: {width:?}");

    //let raw_data = std::fs::read("cat.jpg")?;

    let device = CUDA::new(0)?;

    let image = unsafe { encode_raw_ppm(ppm_img, &device)? };

    Ok(())
}

unsafe fn encode_raw_ppm(
    mut ppm_image: PPMImage,
    device: &CUDA,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut handle: nvjpegHandle_t = null_mut();
    let status = nvjpegCreateSimple(&mut handle);
    check!(status, "Could not create simple handle. ");

    let mut encoder_params: nvjpegEncoderParams_t = null_mut();
    let status = nvjpegEncoderParamsCreate(handle, &mut encoder_params, device.stream().0 as _);
    check!(status, "Could not create encoder params.");

    let mut encoder_state: nvjpegEncoderState_t = null_mut();
    let status = nvjpegEncoderStateCreate(handle, &mut encoder_state, device.stream().0 as _);
    check!(status, "Could not create encoder state.");

    nvjpegEncoderParamsSetSamplingFactors(
        encoder_params,
        nvjpegChromaSubsampling_t_NVJPEG_CSS_420,
        device.stream().0 as _,
    );

    let mut source: nvjpegImage_t = nvjpegImage_t::new();

    source.pitch[0] = ppm_image.width as usize;
    source.pitch[1] = ppm_image.width as usize;
    source.pitch[2] = ppm_image.width as usize;

    let r = Buffer::from((device, ppm_image.r));
    let g = Buffer::from((device, ppm_image.g));
    let b = Buffer::from((device, ppm_image.b));

    source.channel[0] = r.cu_ptr() as *mut _;
    source.channel[1] = g.cu_ptr() as *mut _;
    source.channel[2] = b.cu_ptr() as *mut _;

    println!("source: {source:?}");

    let status = nvjpegEncodeImage(
        handle,
        encoder_state,
        encoder_params,
        &source,
        nvjpegInputFormat_t_NVJPEG_INPUT_RGB,
        ppm_image.width,
        ppm_image.height,
        device.stream().0 as _,
    );
    check!(status, "Could not encode ppm image.");

    device.stream().sync()?;

    let mut len = 0;
    let status = nvjpegEncodeRetrieveBitstream(
        handle,
        encoder_state,
        null_mut(),
        &mut len,
        device.stream().0 as _,
    );
    check!(status, "Cannot retrieve length from bitstream");

    let mut jpeg_data = vec![0u8; len];
    let status = nvjpegEncodeRetrieveBitstream(
        handle,
        encoder_state,
        jpeg_data.as_mut_ptr(),
        &mut len,
        device.stream().0 as _,
    );
    check!(status, "Cannot retrieve data from bitstream");

    std::fs::write("encoded_jpeg.jpg", jpeg_data)?;

    // free resources

    Ok(())
}
