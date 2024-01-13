use std::io::Write;

use fft::{
    num_complex::{Complex, Complex32},
    FftPlanner,
};
use image::{imageops, DynamicImage, ImageBuffer};
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

    let width = wav_spec.sample_rate / FRAMERATE;
    // Eliminate last frame if it's incomplete with integer division
    let height = wav.len() / width;
    let half_width = width / 2 + 1;

    let mut input_samples: Vec<_> = wav
        .into_samples()
        .take((height * width) as usize)
        .map(|r| {
            let r: i16 = r.unwrap();
            Complex::from((r as f32) / -(i16::MIN as f32))
        })
        .collect();

    let mut fft = FftPlanner::<f32>::new();
    fft.plan_fft_forward(width as usize)
        .process(&mut input_samples);

    let max_normal_reducer = |a: f32, b: f32| {
        if !a.is_normal() || a <= b {
            b
        } else {
            a
        }
    };
    let spectrums = || {
        input_samples
            .chunks(width as usize)
            .map(|spectrum| &spectrum[..half_width as usize])
    };
    let sound_intencity_sqrs = || {
        spectrums().flat_map(|spectrum| {
            spectrum
                .iter()
                .enumerate()
                .map(|(f, a)| sound_intencity_sqr(*a, f as f32))
        })
    };

    let sound_sensitivity = std::env::args()
        .nth(2)
        .map(|a| {
            a.parse::<f32>()
                .expect("Could not parse second argument as sound sensitivity (f32)")
        })
        .unwrap_or(0.0)
        - 8.0;
    let norm_factor = 1.0
        / (sound_intencity_sqrs()
            .reduce(max_normal_reducer)
            .unwrap()
            .log2()
            + sound_sensitivity);
    // TODO: use plotters and figure out units
    let img = ImageBuffer::<image::Luma<_>, _>::from_vec(
        half_width,
        height,
        sound_intencity_sqrs()
            .map(|sound_intencity| (sound_intencity.log2() + sound_sensitivity) * norm_factor)
            .map(|l| if l.is_nan() { 0.5 } else { l })
            .collect(),
    )
    .unwrap();

    let img = DynamicImage::from(img).into_rgb8();
    let width = half_width;
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
    let img = imageops::crop_imm(
        &img,
        0,
        empty_frequencies,
        height,
        width - empty_frequencies,
    )
    .to_image();

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

fn sound_intencity_sqr(complex_amplitude: Complex32, frequency: f32) -> f32 {
    complex_amplitude.norm_sqr() * frequency * frequency
}
