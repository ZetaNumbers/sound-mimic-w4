use rustfft::num_complex::{Complex, Complex32};
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
    let wav = hound::WavReader::new(input).unwrap();
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

    let width = wav_spec.sample_rate / FRAMERATE;
    // Eliminate last frame if it's incomplete with integer division
    let height = wav.len() / width;

    let mut input_samples: Vec<_> = wav
        .into_samples()
        .take((height * width) as usize)
        .map(|r| {
            let r: i16 = r.unwrap();
            Complex::from((r as f32) / -(i16::MIN as f32))
        })
        .collect();

    let mut fft = rustfft::FftPlanner::<f32>::new();
    fft.plan_fft_forward(width as usize)
        .process(&mut input_samples);

    let width = width as usize;
    let height = height as usize;

    let tone = |frequency: u32| -> audio::Channel {
        audio::Channel::new(wav_spec.sample_rate, frequency, 1, 100, 0)
    };

    let mut tone_samples = vec![Complex32::new(0.0, 0.0); width];
    let mut scratch = vec![Complex32::new(0.0, 0.0); width];

    #[derive(Clone, Copy, Debug)]
    struct Tone {
        frequency: u32,
        sound_intensity: f32,
    }

    impl Tone {
        fn amplitude(self) -> f32 {
            self.sound_intensity / self.frequency as f32
        }
    }

    let mut best_tones = vec![
        Tone {
            frequency: 0,
            sound_intensity: 0.0
        };
        height
    ];

    // TODO: parallelize?
    for frequency in 20..20000 {
        let mut channel = tone(frequency);
        tone_samples.fill_with(|| {
            let sample = channel.sample_triangle_mono();
            channel.step_sample();
            Complex::new(sample, 0.0)
        });

        fft.plan_fft_forward(width)
            .process_with_scratch(&mut tone_samples, &mut scratch);
        // TODO: normalize tone_samples elements and then entire vector

        for y in 0..height {
            let input_samples = &input_samples[y * width..][..width];
            let best_tone = &mut best_tones[y];

            let sound_intensity = input_samples
                .iter()
                .zip(tone_samples.iter())
                .enumerate()
                .map(|(frequency, (&input_amplitude, &tone_amplitude))| {
                    let frequency = frequency as f32;
                    (sound_intencity_sqr(input_amplitude, frequency)
                        * sound_intencity_sqr(tone_amplitude, frequency))
                    .sqrt()
                })
                .sum::<f32>();

            if best_tone.sound_intensity < sound_intensity {
                *best_tone = Tone {
                    sound_intensity,
                    frequency,
                };
            }
        }
    }

    let inv_max_amplitude = 1.0
        / best_tones
            .iter()
            .map(|t| t.amplitude())
            .reduce(|a, b| a.max(b))
            .unwrap();

    // TODO: output tone CSV table
    let mut writer = tone_stream::Writer::new(std::io::stdout().lock()).unwrap();
    for t in &best_tones {
        let volume = (100.0 * t.amplitude() * inv_max_amplitude)
            .trunc()
            .clamp(0.0, 100.0) as u32;
        writer
            .write_tone(t.frequency, 1, volume, audio::TRIANGLE_CHANNEL)
            .unwrap();
        writer.step_frame().unwrap();
    }
}

fn sound_intencity_sqr(complex_amplitude: Complex32, frequency: f32) -> f32 {
    complex_amplitude.norm_sqr() * frequency * frequency
}
