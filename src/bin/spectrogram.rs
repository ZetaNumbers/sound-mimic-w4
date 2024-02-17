use std::io::Write;

use fft::{num_complex::Complex, FftPlanner};
use image::{imageops, DynamicImage, ImageBuffer};
use nalgebra as na;
use sound_mimic::FRAMERATE;

// TODO: switch to render sound amplitude's logarithm instead of sound intencity level
// WARN: Update README.md documentation if cli documentation below is changed
/// Render spectogram of a sound (WAV file) from stdin into image and output
/// it into stdout. Spectogram highlights sound intensity level, not amplitude.
#[derive(argh::FromArgs)]
struct Spectogram {
    /// format of output image using specified extension
    #[argh(positional, from_str_fn(output_image_format_from_extension))]
    image_format: image::ImageFormat,

    #[argh(positional, default = "0.0")]
    sound_sensitivity: f32,

    /// output image resize width
    #[argh(option, short = 'w')]
    width: Option<u32>,

    /// output image resize height
    #[argh(option, short = 'h')]
    height: Option<u32>,
}

fn output_image_format_from_extension(ext: &str) -> Result<image::ImageFormat, String> {
    image::ImageFormat::from_extension(ext)
        .filter(|fmt| fmt.can_write())
        .ok_or_else(|| format!("unsupported output image format \"{ext:?}\""))
}

fn main() {
    let args: Spectogram = argh::from_env();
    let input = std::io::stdin().lock();
    let wav = hound::WavReader::new(input).unwrap();
    let wav_spec = wav.spec();
    assert_eq!(wav_spec.channels, 1);
    assert_eq!(wav_spec.sample_rate % FRAMERATE, 0);
    assert_eq!(wav_spec.bits_per_sample, 16);
    assert_eq!(wav_spec.sample_format, hound::SampleFormat::Int);

    let width = usize::try_from(wav_spec.sample_rate / FRAMERATE).unwrap();
    let input_samples: Vec<_> = wav
        .into_samples()
        .map(|r| {
            let r: i16 = r.unwrap();
            Complex::from((r as f32) / -(i16::MIN as f32))
        })
        .collect();

    let input_windows = input_samples.windows(width);
    assert!(width >= 1);
    let height = input_windows.len();
    // Since all input samples are on the real axis, one half of
    // its spectrum should be just a conjugate to other one
    let half_width = width / 2 + 1;

    let mut fft_planner = FftPlanner::<f32>::new();
    let fft = fft_planner.plan_fft_forward(width);
    let mut line = na::DVector::zeros(width);
    let mut line_scratch = na::DVector::zeros(fft.get_inplace_scratch_len());

    let mut output = na::DMatrix::zeros(half_width, height);
    assert_eq!(input_windows.len(), height);
    for (input_window, mut output) in input_windows.zip(output.column_iter_mut()) {
        line.copy_from_slice(input_window);
        fft.process_with_scratch(line.as_mut_slice(), line_scratch.as_mut_slice());
        for (input, output) in line.as_slice()[..half_width].iter().zip(&mut output) {
            *output = input.norm_sqr();
        }
    }

    let norm_factor = 1.0
        / (output
            .iter()
            .copied()
            .reduce(max_normal_reducer)
            .unwrap()
            .log2()
            + args.sound_sensitivity);
    output.apply(|sound_aplitude_sqr| {
        let l = (sound_aplitude_sqr.log2() + args.sound_sensitivity) * norm_factor;
        *sound_aplitude_sqr = if l.is_nan() { 0.5 } else { l };
    });

    let width = half_width.try_into().unwrap();
    let height = height.try_into().unwrap();

    // TODO: use plotters and figure out units

    let img =
        ImageBuffer::<image::Luma<_>, _>::from_raw(width, height, output.as_slice().to_owned())
            .unwrap();
    let img = DynamicImage::from(img).to_rgb8();
    // let img = imageops::crop_imm(&img, 0, 0, width, height).to_image();
    assert!(img
        .pixels()
        .all(|px| px.0[0] == px.0[1] && px.0[1] == px.0[2]));
    let img = imageops::rotate270(&img);
    let empty_frequencies = img
        .rows()
        .map(|mut row| row.all(|px| *px == image::Rgb([0, 0, 0])))
        .take_while(|x| *x)
        .count()
        .try_into()
        .unwrap();

    let mut img = imageops::crop_imm(
        &img,
        0,
        empty_frequencies,
        height,
        width - empty_frequencies,
    )
    .to_image();
    if args.width.is_some() || args.height.is_some() {
        img = imageops::resize(
            &img,
            args.width.unwrap_or_else(|| img.width()),
            args.height.unwrap_or_else(|| img.height()),
            imageops::FilterType::CatmullRom,
        )
    }

    let mut out_buf = std::io::Cursor::new(Vec::new());
    image::write_buffer_with_format(
        &mut out_buf,
        &img,
        img.width(),
        img.height(),
        image::ColorType::Rgb8,
        args.image_format,
    )
    .expect("encoding output image");
    std::io::stdout()
        .lock()
        .write_all(out_buf.get_ref())
        .expect("writing output image");
}

fn max_normal_reducer(a: f32, b: f32) -> f32 {
    if !a.is_normal() || a <= b {
        b
    } else {
        a
    }
}
