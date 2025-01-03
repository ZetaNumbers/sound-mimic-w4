use std::{f32::consts::TAU, fs::File, io::BufReader};

use burn::{prelude::*, tensor};
use sound_mimic::{Apu, FRAMERATE, apu, tone_stream};

type Backend = burn::backend::LibTorch;

#[allow(unreachable_code)]
fn device() -> Device<Backend> {
    #[cfg(target_os = "macos")]
    return burn::backend::libtorch::LibTorchDevice::Mps;
    // TODO: Add CUDA
    #[cfg(any(target_os = "windows", target_os = "linux"))]
    return burn::backend::libtorch::LibTorchDevice::Vulkan;
    burn::backend::libtorch::LibTorchDevice::Cpu
}

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

    /// bound tensor memory size (MiB), assuming tensor elements are in `f32`
    #[argh(option, default = "usize::MAX")]
    bound_mem: usize,
}

fn main() {
    let args: Mimic = argh::from_env();

    let Samples {
        samples,
        samples_per_frame,
    } = if args.wav_file == "-" {
        Samples::load_from_wav_reader(std::io::stdin().lock())
    } else {
        Samples::load_from_wav_reader(BufReader::new(
            File::open(args.wav_file).expect("opening wav file"),
        ))
    };
    let device = device();
    let samples = Tensor::from_data(samples, &device);
    let sliding_frames = sliding_frames(samples_per_frame, samples.clone());

    // Crop out frames at the end to rightly compare different sample shifts
    let nframes = sliding_frames.dims()[0] / samples_per_frame;
    let nshifts = nframes * samples_per_frame;

    let sliding_frames = sliding_frames
        .narrow(0, 0, nshifts)
        .reshape([nframes, samples_per_frame, samples_per_frame])
        .swap_dims(0, 1);

    let dft = Dft::new_like(&sliding_frames);
    let offset_frames = dft.clone().apply_sqr(sliding_frames).sqrt();

    let tones = Tones::new(dft);
    let mut best_tones = tones.pick_best_for(offset_frames, args.bound_mem * 1024 * 1024);
    best_tones.scale_to_fit();
    let BestTones {
        scales,
        tone_freq,
        offset,
        error,
    } = best_tones;

    let scales = scales.into_data();
    let scales = scales.iter::<f32>();
    let tone_freq = tone_freq.into_data();
    let tone_freq = tone_freq.iter::<u32>();

    // TODO: log
    eprintln!("Found best tones with error {error:?} at sample offset {offset:?}");

    let mut writer = tone_stream::Writer::new(std::io::stdout().lock()).unwrap();
    for (scale, frequency) in scales.zip(tone_freq) {
        assert!(!scale.is_nan());
        let volume = (100.0 * scale).clamp(0.0, 100.0).trunc() as u32;
        writer
            .write_tone(frequency, 1, volume, apu::TRIANGLE_CHANNEL_FLAG)
            .unwrap();
        writer.step_frame().unwrap();
    }
}

struct BestTones {
    scales: Tensor<Backend, 1>,
    tone_freq: Tensor<Backend, 1, tensor::Int>,
    offset: i64,
    error: f32,
}

impl BestTones {
    fn scale_to_fit(&mut self) {
        let max_scale = self.scales.clone().max().into_scalar();
        assert!(max_scale > 0.0);
        if max_scale > 1.0 {
            return;
        }
        let scale = max_scale.recip();
        self.scales.inplace(|scales| scales.mul_scalar(scale));
    }
}

/// Gets every possible frame offset to pick the best candidate.
///
/// Output of this iterator represents a matrix with samples converted into
/// complex values, while each column represents a samples within one frame.
#[derive(Clone)]
struct Samples {
    samples_per_frame: usize,
    samples: TensorData,
}

impl Samples {
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
        let wav_len = wav.len().try_into().unwrap();
        Samples {
            samples_per_frame: (wav_spec.sample_rate / FRAMERATE).try_into().unwrap(),
            samples: TensorData::new::<f32, _>(
                wav.into_samples::<i16>()
                    .map(|s| {
                        (s.expect("error while reading wav file") as f32 / -(i16::MIN as f32))
                            .into()
                    })
                    .collect(),
                [wav_len],
            ),
        }
    }
}

// return: shift * frame_sample
/// Rearrange samples as sliding frame samples.
fn sliding_frames(samples_per_frame: usize, samples: Tensor<Backend, 1>) -> Tensor<Backend, 2> {
    let [nsamples] = samples.dims();

    let nshifts = nsamples + 1 - samples_per_frame;
    let device = samples.device();
    let frame_sample_indices = Tensor::arange(0..samples_per_frame.try_into().unwrap(), &device);
    let shift_indices = Tensor::arange(0..i64::try_from(nshifts).unwrap(), &device);
    let indices = shift_indices
        .unsqueeze_dim::<2>(1)
        .repeat_dim(1, samples_per_frame)
        .add(frame_sample_indices.unsqueeze_dim(0).repeat_dim(0, nshifts))
        .flatten(0, 1);
    samples
        .gather(0, indices)
        .reshape([nshifts, samples_per_frame])
}

/// Real valued DFT.
#[derive(Clone)]
struct Dft {
    cos: Tensor<Backend, 2>,
    sin: Tensor<Backend, 2>,
}

impl Dft {
    fn new(samples_per_frame: usize, device: &Device<Backend>) -> Self {
        let freq_ncols = samples_per_frame;
        let freq_nrows = samples_per_frame / 2;

        let cols =
            Tensor::<Backend, 1, _>::arange(0..freq_ncols.try_into().unwrap(), device).float();
        let rows = Tensor::arange(1..(freq_nrows + 1).try_into().unwrap(), device).float();

        let freqs = cols.unsqueeze_dim::<2>(1).repeat_dim(1, freq_nrows).mul(
            rows.mul_scalar(TAU / samples_per_frame as f32)
                .unsqueeze_dim(0)
                .repeat_dim(0, freq_ncols),
        );

        Dft {
            cos: freqs.clone().cos(),
            sin: freqs.sin(),
        }
    }

    fn new_like<const D: usize>(sliding_frames: &Tensor<Backend, D>) -> Self {
        Dft::new(
            *sliding_frames.dims().last().unwrap(),
            &sliding_frames.device(),
        )
    }

    fn device(&self) -> Device<Backend> {
        self.cos.device()
    }

    fn samples_per_frame(&self) -> usize {
        self.cos.dims()[0]
    }

    // frames: ... * frame_sample
    // return: ... * freq
    fn apply_sqr<const D: usize>(self, frames: Tensor<Backend, D>) -> Tensor<Backend, D> {
        assert!(D >= 2);

        let mut repeat_dims = frames.dims();
        *repeat_dims.last_chunk_mut::<2>().unwrap() = [1, 1];

        let new_axes = frames.dims().map({
            let mut i = 0;
            move |_| {
                let t = i;
                i += 1;
                t
            }
        });
        // Crop out two last elements
        let new_axes = &new_axes[0..D - 2];

        let zero_freq = frames.clone().sum_dim(D - 1).mul_scalar(0.5);

        // transposed matrix multiplication to reverse order
        let cos = frames
            .clone()
            .matmul(self.cos.unsqueeze_dims(new_axes).repeat(&repeat_dims));
        let sin = frames.matmul(self.sin.unsqueeze_dims(new_axes).repeat(&repeat_dims));

        let const_amplitude = zero_freq.powi_scalar(2);
        let wave_amplitude = cos.powi_scalar(2).add(sin.powi_scalar(2));
        Tensor::cat(vec![const_amplitude, wave_amplitude], D - 1)
    }
}

fn tone(apu: &mut Apu, frequency: u32, dest: &mut [f32]) {
    debug_assert_eq!(dest.len() * 60, usize::try_from(apu.sample_rate()).unwrap());
    apu.tone(frequency, 1, 100, apu::TRIANGLE_CHANNEL_FLAG);
    dest.fill_with(|| apu.next().unwrap()[0] as f32 / u16::MAX as f32);
    apu.tick();
}

#[derive(Clone)]
struct Tones {
    // norm_spectrums: tone_freq * freq
    norm_spectrums: Tensor<Backend, 2>,
    // original_norm: tone_freq
    original_norm: Tensor<Backend, 1>,
}

impl Tones {
    fn new(dft: Dft) -> Self {
        let samples_per_frame = dft.samples_per_frame();
        let mut apu = Apu::new(u32::try_from(samples_per_frame).unwrap() * FRAMERATE);
        // TODO: use tensor of frequencies
        let ntone_freq = (MAX_FREQUENCY - MIN_FREQUENCY).try_into().unwrap();
        let mut tones_samples = vec![0.0; samples_per_frame * ntone_freq];
        for frequency in MIN_FREQUENCY..MAX_FREQUENCY {
            let shift = usize::try_from(frequency - MIN_FREQUENCY).unwrap();
            tone(
                &mut apu,
                frequency,
                &mut tones_samples[(shift * samples_per_frame)..((shift + 1) * samples_per_frame)],
            );
        }

        // tone_freq * frame_sample
        let tones_samples = Tensor::from_data(
            TensorData::new(tones_samples, [ntone_freq, samples_per_frame]),
            &dft.device(),
        );

        // tone_freq * freq
        let spectrums_sqr = dft.apply_sqr(tones_samples);
        let original_norm = spectrums_sqr.clone().sum_dim(1).sqrt();
        let norm_spectrums = spectrums_sqr
            .sqrt()
            .div(original_norm.clone().repeat_dim(1, original_norm.dims()[1]));

        Tones {
            norm_spectrums,
            original_norm: original_norm.squeeze(1),
        }
    }

    // offset_frame_chunks: offset * chunk_frame * freq
    /// Bound memory usage by `bound_mem` bytes
    fn pick_best_for(self, offset_frames: Tensor<Backend, 3>, bound_mem: usize) -> BestTones {
        let [noffsets, nframes, nfreq] = offset_frames.dims();
        let [ntone_freq, _] = self.norm_spectrums.dims();
        let trans_norm_spectrums = self.norm_spectrums.clone().transpose();

        let mem_estimate = noffsets
            .checked_mul(nframes)
            .and_then(|s| s.checked_mul(ntone_freq))
            .and_then(|s| s.checked_mul(nfreq))
            .and_then(|s| s.checked_mul(4)) // 4 bytes of f32
            .unwrap();
        let splits = mem_estimate.div_ceil(bound_mem);

        let (offset_chunks, frame_chunks) = if noffsets > splits {
            assert!(noffsets * nframes >= splits, "too little memory");
            (noffsets, splits.div_ceil(noffsets).div_ceil(nframes))
        } else {
            (splits.div_ceil(noffsets), 1)
        };

        let offset_frame_chunks = offset_frames.chunk(offset_chunks, 0);

        let mut optimal_tone_idx_by_offset_chunk = Vec::with_capacity(offset_frame_chunks.len());
        let mut error_by_offset_chunk = Vec::with_capacity(offset_frame_chunks.len());
        let mut scales_by_offset_chunk = Vec::with_capacity(offset_frame_chunks.len());

        let progress = indicatif::ProgressBar::new(offset_frame_chunks.len().try_into().unwrap());

        for offset_frames in offset_frame_chunks {
            let offset_frame_chunks = offset_frames.chunk(frame_chunks, 1);

            let mut optimal_tone_idx_by_frame_chunk = Vec::with_capacity(offset_frame_chunks.len());
            let mut error_by_frame_chunk = Vec::with_capacity(offset_frame_chunks.len());
            let mut scales_by_frame_chunk = Vec::with_capacity(offset_frame_chunks.len());

            for offset_frames in offset_frame_chunks {
                let [noffsets, nframes, _] = offset_frames.dims();

                // scales: chunk_offset * chunk_frame * tone_freq
                let scales = offset_frames.clone().matmul(
                    trans_norm_spectrums
                        .clone()
                        .unsqueeze_dim(0)
                        .repeat_dim(0, noffsets),
                );
                // tone_spectrums_scaled: chunk_offset * chunk_frame * tone_freq * freq
                let tone_spectrums_scaled = self
                    .norm_spectrums
                    .clone()
                    .unsqueeze_dims::<4>(&[0, 1])
                    .repeat(&[noffsets, nframes, 1, 1])
                    .mul(scales.clone().unsqueeze_dim::<4>(3).repeat_dim(3, nfreq));
                // TODO: try a-weighted error
                // error: chunk_offset * chunk_frame * tone_freq
                let error = offset_frames
                    .unsqueeze_dim::<4>(2)
                    .repeat_dim(2, ntone_freq)
                    .sub(tone_spectrums_scaled)
                    .powi_scalar(2)
                    .sum_dim(3)
                    .squeeze::<3>(3);

                // error: chunk_offset * chunk_frame * 1
                // optimal_tone_idx: chunk_offset * chunk_frame * 1
                let (error, optimal_tone_idx) = error.min_dim_with_indices(2);

                // scales: chunk_offset * chunk_frame * 1
                let scales = scales.gather(2, optimal_tone_idx.clone());

                scales_by_frame_chunk.push(scales);
                error_by_frame_chunk.push(error);
                optimal_tone_idx_by_frame_chunk.push(optimal_tone_idx);
            }

            let error = Tensor::cat(error_by_frame_chunk, 1);
            let optimal_tone_idx = Tensor::cat(optimal_tone_idx_by_frame_chunk, 1);
            let scales = Tensor::cat(scales_by_frame_chunk, 1);

            scales_by_offset_chunk.push(scales);
            error_by_offset_chunk.push(error);
            optimal_tone_idx_by_offset_chunk.push(optimal_tone_idx);
            progress.inc(1);
        }
        progress.finish_with_message("Traversed all tones");

        let error = Tensor::cat(error_by_offset_chunk, 0);
        let optimal_tone_idx = Tensor::cat(optimal_tone_idx_by_offset_chunk, 0);
        let scales = Tensor::cat(scales_by_offset_chunk, 0);

        // error: 1 * 1 * 1
        // optimal_offset: 1 * 1 * 1
        let (error, optimal_offset) = error.sum_dim(1).min_dim_with_indices(0);
        let offset = optimal_offset.into_scalar();
        // optimal_tone_idx: frame * 1
        let optimal_tone_idx = optimal_tone_idx
            .narrow(0, offset.try_into().unwrap(), 1)
            .squeeze::<2>(0);
        // scales: frame
        let scales = scales
            .slice([Some((offset, offset + 1)), None, None])
            .squeeze::<2>(0)
            .div(self.original_norm.unsqueeze_dim(0).repeat_dim(0, nframes))
            .gather(1, optimal_tone_idx.clone())
            .squeeze::<1>(1);
        // tone_freq: frame
        let tone_freq = optimal_tone_idx.squeeze::<1>(1).add_scalar(MIN_FREQUENCY);
        let error = error.into_scalar();
        BestTones {
            scales,
            tone_freq,
            offset,
            error,
        }
    }
}
