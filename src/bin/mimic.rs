use std::io::Write as _;

use rustfft::num_complex::{Complex, Complex32};
use sound_mimic::{audio, FRAMERATE};

fn main() {
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
    for frequency in 20..20000 {
        let mut channel = tone(frequency);
        tone_samples.fill_with(|| {
            let sample = channel.sample_pulse_mono();
            channel.step_sample();
            Complex::new(sample, 0.0)
        });

        fft.plan_fft_forward(width)
            .process_with_scratch(&mut tone_samples, &mut scratch);

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

    tone_samples = vec![Complex32::new(0.0, 0.0); width * height];
    let mut phase = 0.0;
    for (y, &t) in best_tones.iter().enumerate() {
        let volume = (100.0 * t.amplitude() * inv_max_amplitude)
            .trunc()
            .clamp(0.0, 100.0) as u32;
        let frequency = t.frequency;
        eprintln!("Tone {{ frequency: {frequency:5}, volume: {volume:3} }}");
        let mut channel =
            audio::Channel::with_phase(wav_spec.sample_rate, frequency, 1, volume, 0, phase);
        for tone_sample in &mut tone_samples[y * width..][..width] {
            *tone_sample = channel.sample_pulse_mono().into();
            channel.step_sample();
        }
        phase = channel.phase;
    }

    output_audio(tone_samples, wav_spec);
}

fn sound_intencity_sqr(complex_amplitude: Complex32, frequency: f32) -> f32 {
    complex_amplitude.norm_sqr() * frequency * frequency
}

fn output_audio(samples: Vec<Complex<f32>>, wav_spec: hound::WavSpec) {
    let mut output = std::io::Cursor::new(Vec::new());
    let mut wav = hound::WavWriter::new(&mut output, wav_spec).unwrap();
    for sample in samples {
        wav.write_sample(
            (sample.re * -(i16::MIN as f32))
                .round()
                .clamp(i16::MIN as f32, i16::MAX as f32) as i16,
        )
        .unwrap();
    }
    wav.finalize().unwrap();
    std::io::stdout()
        .lock()
        .write_all(output.get_ref())
        .unwrap();
}
