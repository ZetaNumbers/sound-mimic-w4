use std::ops::{MulAssign, Range};

use nalgebra as na;
use ordered_float::NotNan;
use rayon::prelude::*;
use rustfft::{
    num_complex::{Complex, Complex32},
    num_traits::Zero,
    FftPlanner,
};
use sound_mimic::{audio, tone_stream, FRAMERATE};

// WARN: Update README.md documentation if cli documentation below is changed
// TODO: Option to choose wave channels and their settings
/// Chooses most suitable tones for each frame of input sound (wav file)
/// and outputs these tones as CSV table in stdout.
#[derive(argh::FromArgs)]
struct Mimic {}

fn main() {
    let _args: Mimic = argh::from_env();

    let input = std::io::stdin().lock();
    let sliding_frames = ComplexSamplesInSlidingFrames::load_from_wav_reader(input);

    let best_mimic_tones = sliding_frames.pick_best_mimic_tones();

    let mut writer = tone_stream::Writer::new(std::io::stdout().lock()).unwrap();
    for tone in &best_mimic_tones.frames {
        let volume = (100.0 * tone.scale.into_inner()).clamp(0.0, 100.0).trunc() as u32;
        writer
            .write_tone(tone.frequency, 1, volume, audio::TRIANGLE_CHANNEL)
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

/// Gets every possible frame offset to pick the best candidate.
///
/// Output of this iterator represents a matrix with samples converted into
/// complex values, while each column represents a samples within one frame.
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

    #[inline(never)]
    fn pick_best_mimic_tones(self) -> BestTones {
        struct Context {
            fft_planner: FftPlanner<f32>,
            lower_conjugate_rows: usize,
            conjugate_rows: Range<usize>,
        }

        let mut fft_planner = FftPlanner::<f32>::new();

        let samples_per_frame = self.samples_per_frame;
        let lower_nonconjugate_nrows = samples_per_frame / 2 + 1;
        let conjugate_rows = 1..(samples_per_frame + 1) / 2;
        let forward_fft = fft_planner.plan_fft_forward(samples_per_frame);
        let fft_descale = fourier_scale_factor(samples_per_frame);

        let tone = |frequency: u32| -> audio::Channel {
            audio::Channel::new(samples_per_frame as u32 * FRAMERATE, frequency, 1, 100, 0)
        };

        let mut best_mimic_tones = self
            .par_bridge()
            .map_init(
                || {
                    (
                        vec![Complex32::zero(); forward_fft.get_inplace_scratch_len()],
                        na::DVector::zeros(samples_per_frame),
                        na::DVector::zeros(lower_nonconjugate_nrows),
                        na::DVector::zeros(lower_nonconjugate_nrows),
                    )
                },
                |(fft_scratch, tone_samples, tone_spectrum, tone_spectrum_scaled), mut frames| {
                    forward_fft.process_with_scratch(frames.as_mut_slice(), fft_scratch);

                    let frames = {
                        let fft_descale = fourier_scale_factor(samples_per_frame);
                        let mut frames =
                            frames.rows(0, lower_nonconjugate_nrows).map(Complex::norm);
                        frames.row_mut(0).mul_assign(fft_descale);
                        frames
                            .rows_range_mut(conjugate_rows.clone())
                            .mul_assign(2.0 * fft_descale);
                        if samples_per_frame % 2 == 0 {
                            frames
                                .row_mut(lower_nonconjugate_nrows - 1)
                                .mul_assign(fft_descale);
                        }
                        frames
                    };

                    let mut best_tones = BestTones::new(frames.ncols());

                    // TODO: parallelize?
                    for frequency in 20..20000 {
                        let mut channel = tone(frequency);
                        tone_samples.apply(|dest| {
                            let sample = channel.sample_triangle_mono();
                            channel.step_sample();
                            *dest = Complex::new(sample, 0.0)
                        });
                        forward_fft.process_with_scratch(tone_samples.as_mut_slice(), fft_scratch);
                        let tone_spectrum_view = tone_samples.rows(0, lower_nonconjugate_nrows);
                        assert_eq!(tone_spectrum.shape(), tone_spectrum_view.shape());
                        tone_spectrum
                            .zip_apply(&tone_spectrum_view, |dest, src| *dest = src.norm());
                        {
                            tone_spectrum[0] *= fft_descale;
                            tone_spectrum
                                .rows_range_mut(conjugate_rows.clone())
                                .mul_assign(2.0 * fft_descale);
                            if samples_per_frame % 2 == 0 {
                                tone_spectrum[lower_nonconjugate_nrows - 1] *= fft_descale;
                            }
                        }
                        let original_descale = NotNan::new(tone_spectrum.norm().recip()).unwrap();
                        *tone_spectrum *= original_descale.into_inner();
                        // ^ we pick the best orthonormal basis of one vector

                        for y in 0..frames.ncols() {
                            let frame = frames.column(y);
                            let best_tone = &mut best_tones.frames[y];

                            let scale = NotNan::new(frame.dot(tone_spectrum)).unwrap();
                            tone_spectrum_scaled.copy_from(tone_spectrum);
                            tone_spectrum_scaled.scale_mut(scale.into_inner());

                            let error =
                                NotNan::new(tone_spectrum_scaled.metric_distance(&frame)).unwrap();

                            if best_tone.error > error {
                                *best_tone = BestTone {
                                    scale: scale * original_descale,
                                    frequency,
                                    error,
                                };
                            }
                        }
                    }
                    best_tones.eval_total_error();
                    best_tones
                },
            )
            .min_by_key(|t| t.total_error)
            .expect("input sound has not enough samples");
        best_mimic_tones.scale_to_fit();
        best_mimic_tones
    }
}

impl Iterator for ComplexSamplesInSlidingFrames {
    type Item = na::DMatrix<Complex32>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= FRAMERATE as usize {
            return None;
        }
        let samples = &self.samples[self.offset..];
        // Eliminate last frame if it's incomplete with integer division
        let frames = samples.len() / self.samples_per_frame;
        if frames < 1 {
            return None;
        }
        let samples = &samples[..frames * self.samples_per_frame];
        let out = na::DMatrix::from_column_slice(self.samples_per_frame, frames, samples);
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
    use super::{fourier_scale_factor, na};
    use rustfft::num_complex::Complex32;

    #[test]
    fn spectral_density_scale_factor() {
        const SAMPLE_COUNT: usize = 1024;

        let mut fft_planner = rustfft::FftPlanner::new();
        let mut v = na::DVector::from_fn(SAMPLE_COUNT, |i, _| {
            let t = i as f32 * std::f32::consts::TAU / SAMPLE_COUNT as f32;
            Complex32::from_polar(1.0, t)
        });

        let prenorm = v.norm();
        let scale_factor = fourier_scale_factor(SAMPLE_COUNT);

        let fft = fft_planner.plan_fft_forward(SAMPLE_COUNT);
        fft.process(v.as_mut_slice());
        v.scale_mut(scale_factor);
        let postnorm = v.norm();
        assert!((prenorm - postnorm) < std::f32::EPSILON);

        let fft = fft_planner.plan_fft_inverse(SAMPLE_COUNT);
        fft.process(v.as_mut_slice());
        v.scale_mut(scale_factor);
        let preprenorm = v.norm();
        assert!((preprenorm - prenorm) < std::f32::EPSILON);
    }
}
