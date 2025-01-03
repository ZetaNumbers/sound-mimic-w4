use std::{
    fs::File,
    io::BufReader,
    ops::{MulAssign, Range},
    sync::Arc,
};

use fft::{Fft, FftPlanner, num_complex::Complex32, num_traits::Zero};
use indicatif::ProgressBar;
use ndarray::{Array1, Array2, Axis, azip, s};
use ndarray_linalg::Norm;
use ndarray_stats::DeviationExt;
use ordered_float::NotNan;
use rayon::prelude::*;
use sound_mimic::{Apu, FRAMERATE, apu, tone_stream};

const MIN_FREQUENCY: u32 = 20;
const MAX_FREQUENCY: u32 = 20000;

// WARN: Update README.md documentation if cli documentation below is changed
// TODO: Option to choose wave channels and their settings
/// Chooses most suitable tones for each frame of input sound (wav file)
/// and outputs these tones as CSV table in stdout.
#[derive(argh::FromArgs)]
struct Mimic {
    /// input wav file path or `-` to read from stdin
    #[argh(positional)]
    wav_file: String,
}

fn main() {
    let args: Mimic = argh::from_env();

    let sliding_frames = if args.wav_file == "-" {
        ComplexSamplesInSlidingFrames::load_from_wav_reader(std::io::stdin().lock())
    } else {
        ComplexSamplesInSlidingFrames::load_from_wav_reader(BufReader::new(
            File::open(args.wav_file).expect("opening wav file"),
        ))
    };

    let best_mimic_tones = sliding_frames.pick_best_mimic_tones();

    let mut writer = tone_stream::Writer::new(std::io::stdout().lock()).unwrap();
    for tone in &best_mimic_tones.frames {
        let volume = (100.0 * tone.scale.into_inner()).clamp(0.0, 100.0).trunc() as u32;
        writer
            .write_tone(tone.frequency, 1, volume, apu::TRIANGLE_CHANNEL_FLAG)
            .unwrap();
        writer.step_frame().unwrap();
    }
}

#[derive(Clone, Copy)]
struct BestTone {
    frequency: u32,
    scale: NotNan<f32>,
    error: NotNan<f32>,
}

struct BestTones {
    frames: Vec<BestTone>,
    total_error: NotNan<f32>,
}

impl BestTones {
    fn new(frames: usize) -> Self {
        BestTones {
            frames: vec![
                BestTone {
                    frequency: 0,
                    scale: NotNan::new(0.0).unwrap(),
                    error: NotNan::new(f32::INFINITY).unwrap(),
                };
                frames
            ],
            total_error: NotNan::new(f32::INFINITY).unwrap(),
        }
    }

    fn eval_total_error(&mut self) {
        self.total_error = self.frames.iter().map(|t| t.error).sum();
    }

    fn max_scale(&self) -> Option<f32> {
        self.frames
            .iter()
            .map(|t| t.scale)
            .max()
            .map(|f| f.into_inner())
    }

    fn scale_to_fit(&mut self) {
        let Some(max_scale) = self.max_scale().filter(|s| *s > 1.0) else {
            return;
        };
        let scale = max_scale.recip();
        self.frames.iter_mut().for_each(|t| t.scale *= scale);
    }
}

fn tone(apu: &mut Apu, frequency: u32, dest: &mut [Complex32]) {
    debug_assert_eq!(dest.len() * 60, usize::try_from(apu.sample_rate()).unwrap());
    apu.tone(frequency, 1, 100, apu::TRIANGLE_CHANNEL_FLAG);
    dest.fill_with(|| (apu.next().unwrap()[0] as f32 / u16::MAX as f32).into());
    apu.tick();
}

/// Gets every possible frame offset to pick the best candidate.
///
/// Output of this iterator represents a matrix with samples converted into
/// complex values, while each row represents a samples within one frame.
struct ComplexSamplesInSlidingFrames {
    offset: usize,
    samples_per_frame: usize,
    samples: Vec<Complex32>,
}

impl ComplexSamplesInSlidingFrames {
    fn load_from_wav_reader<R>(reader: R) -> Self
    where
        R: std::io::Read,
    {
        let wav = hound::WavReader::new(reader).unwrap();
        let wav_spec = wav.spec();
        // TODO: stereo support?
        assert_eq!(wav_spec.channels, 1, "input sound has to have one channel");
        assert_eq!(
            wav_spec.sample_rate % FRAMERATE,
            0,
            "input sound has to have sample rate divisible by framerate ({FRAMERATE})",
        );
        assert_eq!(
            (wav_spec.sample_format, wav_spec.bits_per_sample),
            (hound::SampleFormat::Int, 16),
            "input sound has to have samples of type i16"
        );
        ComplexSamplesInSlidingFrames {
            offset: 0,
            samples_per_frame: (wav_spec.sample_rate / FRAMERATE).try_into().unwrap(),
            samples: wav
                .into_samples::<i16>()
                .map(|s| {
                    (s.expect("error while reading wav file") as f32 / -(i16::MIN as f32)).into()
                })
                .collect(),
        }
    }

    fn pick_best_mimic_tones(self) -> BestTones {
        struct Context {
            samples_per_frame: usize,
            lower_nonconjugate_ncols: usize,
            conjugate_rows: Range<usize>,
            forward_fft: Arc<dyn Fft<f32>>,
            fft_descale: f32,
            tone_spectrums: Array2<f32>,
            original_descale: Vec<NotNan<f32>>,
            progress_bar: ProgressBar,
        }

        impl Context {
            fn new(samples_per_frame: usize) -> Self {
                let mut fft_planner = FftPlanner::<f32>::new();
                let lower_nonconjugate_nrows = samples_per_frame / 2 + 1;
                let mut out = Context {
                    conjugate_rows: 1..(samples_per_frame + 1) / 2,
                    forward_fft: fft_planner.plan_fft_forward(samples_per_frame),
                    fft_descale: fourier_scale_factor(samples_per_frame),
                    tone_spectrums: Array2::zeros([
                        lower_nonconjugate_nrows,
                        (MAX_FREQUENCY - MIN_FREQUENCY).try_into().unwrap(),
                    ]),
                    original_descale: Vec::new(),
                    progress_bar: ProgressBar::new(samples_per_frame.try_into().unwrap()),
                    lower_nonconjugate_ncols: lower_nonconjugate_nrows,
                    samples_per_frame,
                };
                out.tone_bases_init();
                out
            }

            fn local_context(&self) -> LocalContext<'_> {
                LocalContext {
                    cx: self,
                    fft_scratch: vec![
                        Complex32::zero();
                        self.forward_fft.get_inplace_scratch_len()
                    ],
                    tone_spectrum_scaled: Array1::zeros(self.lower_nonconjugate_ncols),
                }
            }

            #[cfg_attr(feature = "profile", inline(never))]
            fn tone_bases_init(&mut self) {
                self.original_descale = self
                    .tone_spectrums
                    .axis_iter_mut(Axis(1))
                    .into_par_iter()
                    .enumerate()
                    .map_init(
                        || {
                            (
                                vec![Complex32::zero(); self.forward_fft.get_inplace_scratch_len()],
                                Array1::zeros(self.samples_per_frame),
                                Apu::new(self.samples_per_frame as u32 * FRAMERATE),
                            )
                        },
                        |(fft_scratch, tone_samples, apu), (column_idx, mut tone_spectrum)| {
                            let frequency = MIN_FREQUENCY + u32::try_from(column_idx).unwrap();
                            tone(apu, frequency, tone_samples.as_slice_mut().unwrap());
                            self.forward_fft.process_with_scratch(
                                tone_samples.as_slice_mut().unwrap(),
                                fft_scratch,
                            );
                            let tone_spectrum_view =
                                tone_samples.slice(s![..self.lower_nonconjugate_ncols]);
                            assert_eq!(tone_spectrum.shape(), tone_spectrum_view.shape());
                            azip!((dest in &mut tone_spectrum, &src in &tone_spectrum_view) *dest = src.norm());
                            {
                                tone_spectrum[0] *= self.fft_descale;
                                tone_spectrum
                                    .slice_mut(s![self.conjugate_rows.clone()])
                                    .mul_assign(2.0 * self.fft_descale);
                                if self.samples_per_frame % 2 == 0 {
                                    tone_spectrum[self.lower_nonconjugate_ncols - 1] *=
                                        self.fft_descale;
                                }
                            }
                            let original_descale =
                                NotNan::new(tone_spectrum.norm_l2().recip()).unwrap();
                            tone_spectrum *= original_descale.into_inner();
                            // ^ we pick the best orthonormal basis of one vector
                            original_descale
                        },
                    )
                    .collect();
            }
        }

        struct LocalContext<'a> {
            cx: &'a Context,
            fft_scratch: Vec<Complex32>,
            tone_spectrum_scaled: Array1<f32>,
        }

        impl LocalContext<'_> {
            fn pick_best_mimic_tones(&mut self, mut frames: Array2<Complex32>) -> BestTones {
                self.cx
                    .forward_fft
                    .process_with_scratch(frames.as_slice_mut().unwrap(), &mut self.fft_scratch);

                let frames = {
                    let fft_descale = fourier_scale_factor(self.cx.samples_per_frame);
                    let mut frames = frames
                        .slice(s![..self.cx.lower_nonconjugate_ncols, ..])
                        .map(|c| c.norm());
                    frames.row_mut(0).mul_assign(fft_descale);
                    frames
                        .slice_mut(s![self.cx.conjugate_rows.clone(), ..])
                        .mul_assign(2.0 * self.cx.fft_descale);
                    if self.cx.samples_per_frame % 2 == 0 {
                        frames
                            .column_mut(self.cx.lower_nonconjugate_ncols - 1)
                            .mul_assign(fft_descale);
                    }
                    frames
                };

                let mut best_tones = BestTones::new(frames.ncols());

                self.pick_best_mimic_tones_impl(frames, &mut best_tones);
                best_tones.eval_total_error();
                best_tones
            }

            #[cfg_attr(feature = "profile", inline(never))]
            fn pick_best_mimic_tones_impl(
                &mut self,
                frames: Array2<f32>,
                best_tones: &mut BestTones,
            ) {
                self.cx
                    .tone_spectrums
                    .axis_iter(Axis(0))
                    .zip(&self.cx.original_descale)
                    .enumerate()
                    .for_each(
                        |(tone_spectrum_column_idx, (tone_spectrum, original_descale))| {
                            let frequency =
                                MIN_FREQUENCY + u32::try_from(tone_spectrum_column_idx).unwrap();
                            frames
                                .axis_iter(Axis(0))
                                .zip(&mut best_tones.frames)
                                .for_each(|(frame, best_tone)| {
                                    let scale = frame.dot(&tone_spectrum);
                                    let scale = NotNan::new(scale).unwrap();
                                    self.tone_spectrum_scaled.assign(&tone_spectrum);
                                    self.tone_spectrum_scaled *= scale.into_inner();

                                    let error =
                                        self.tone_spectrum_scaled.sq_l2_dist(&frame).unwrap();
                                    let error = NotNan::new(error).unwrap();

                                    if best_tone.error > error {
                                        *best_tone = BestTone {
                                            scale: scale * original_descale,
                                            frequency,
                                            error,
                                        };
                                    }
                                });
                        },
                    );
                self.cx.progress_bar.inc(1);
            }
        }

        let cx = Context::new(self.samples_per_frame);

        let mut best_mimic_tones = self
            .par_bridge()
            .map_init(
                || cx.local_context(),
                |lcx, frames| lcx.pick_best_mimic_tones(frames),
            )
            .min_by_key(|t| t.total_error)
            .expect("input sound has not enough samples");
        best_mimic_tones.scale_to_fit();
        best_mimic_tones
    }
}

impl Iterator for ComplexSamplesInSlidingFrames {
    type Item = Array2<Complex32>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.samples_per_frame {
            return None;
        }
        let samples = &self.samples[self.offset..];
        // Eliminate last frame if it's incomplete with integer division
        let frames = samples.len() / self.samples_per_frame;
        if frames < 1 {
            return None;
        }
        let samples = &samples[..frames * self.samples_per_frame];
        let out =
            Array2::from_shape_vec([self.samples_per_frame, frames], samples.to_vec()).unwrap();
        self.offset += 1;
        Some(out)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(FRAMERATE as usize - self.offset - 1))
    }
}

fn fourier_scale_factor_sqr(len: usize) -> f32 {
    (len as f32).recip()
}

fn fourier_scale_factor(len: usize) -> f32 {
    fourier_scale_factor_sqr(len).sqrt()
}

#[cfg(test)]
mod tests {
    use super::fourier_scale_factor;
    use fft::{FftPlanner, num_complex::Complex32};
    use ndarray::Array1;
    use ndarray_linalg::norm::Norm;

    #[test]
    fn spectral_density_scale_factor() {
        const SAMPLE_COUNT: usize = 1024;

        let mut fft_planner = FftPlanner::new();
        let mut v: Array1<Complex32> = (0..SAMPLE_COUNT)
            .map(|i| {
                let t = i as f32 * std::f32::consts::TAU / SAMPLE_COUNT as f32;
                Complex32::from_polar(1.0, t)
            })
            .collect();

        let prenorm = Norm::norm_l2(&v);
        let scale_factor = fourier_scale_factor(SAMPLE_COUNT);

        let fft = fft_planner.plan_fft_forward(SAMPLE_COUNT);
        fft.process(v.as_slice_mut().unwrap());
        v *= Complex32::from(scale_factor);
        let postnorm = v.norm();
        assert!((prenorm - postnorm) < std::f32::EPSILON);

        let fft = fft_planner.plan_fft_inverse(SAMPLE_COUNT);
        fft.process(v.as_slice_mut().unwrap());
        v *= Complex32::from(scale_factor);
        let preprenorm = v.norm();
        assert!((preprenorm - prenorm) < std::f32::EPSILON);
    }
}
