use std::io::Write;

use sound_mimic::{audio, tone_stream, Audio, FRAMERATE};

pub const DEFAULT_SAMPLES_PER_FRAME: u32 = 735;
pub const DEFAULT_SAMPLE_RATE: u32 = DEFAULT_SAMPLES_PER_FRAME * FRAMERATE;

// WARN: Update README.md documentation if cli documentation below is changed
/// Transforms tone CSV table from stdin into WAV file and outputs it into stdout.
#[derive(argh::FromArgs)]
struct TonePlayback {
    /// ignore sound panning and output mono sound
    #[argh(switch)]
    mono: bool,
    /// sound sample rate, must be divisible by 60 (framerate)
    #[argh(option, default = "DEFAULT_SAMPLE_RATE")]
    sample_rate: u32,
}

fn main() {
    let args: TonePlayback = argh::from_env();
    assert_eq!(
        args.sample_rate % FRAMERATE,
        0,
        "sound sample rate must be divisible by 60 (framerate)"
    );

    let tones = tone_stream::Reader::new(std::io::stdin().lock()).unwrap();
    let mut tones: Vec<_> = tones.collect::<Result<_, _>>().unwrap();
    tones.sort_by_key(|t| t.frame);

    let mut audio = Audio::new(args.sample_rate);

    let mut output = std::io::Cursor::new(Vec::new());
    let mut wav = hound::WavWriter::new(
        &mut output,
        hound::WavSpec {
            channels: if args.mono { 1 } else { 2 },
            sample_rate: args.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();

    let samples_per_frame = args.sample_rate / FRAMERATE;
    let mut frame = 0;
    if args.mono {
        for t in tones {
            while frame != t.frame {
                for _ in 0..samples_per_frame {
                    let sample = audio.sample_mono();
                    audio.step_sample();
                    wav.write_sample(sample).unwrap();
                }
                frame += 1;
            }

            audio.tone(t.frequency, t.duration, t.volume, t.flags);
        }
        while !audio.has_ended() {
            for _ in 0..samples_per_frame {
                let sample = audio.sample_mono();
                audio.step_sample();
                wav.write_sample(sample).unwrap();
            }
        }
    } else {
        for t in tones {
            while frame != t.frame {
                for _ in 0..samples_per_frame {
                    let audio::StereoSample { left, right } = audio.sample();
                    audio.step_sample();
                    wav.write_sample(left).unwrap();
                    wav.write_sample(right).unwrap();
                }
                frame += 1;
            }

            audio.tone(t.frequency, t.duration, t.volume, t.flags);
        }
        while !audio.has_ended() {
            for _ in 0..samples_per_frame {
                let audio::StereoSample { left, right } = audio.sample();
                audio.step_sample();
                wav.write_sample(left).unwrap();
                wav.write_sample(right).unwrap();
            }
        }
    };

    wav.finalize().unwrap();
    std::io::stdout()
        .lock()
        .write_all(output.get_ref())
        .unwrap();
}
