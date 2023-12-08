use std::io::Write;

use sound_mimic::{audio, tone_stream, Audio, FRAMERATE};

const SAMPLES_PER_FRAME: u32 = 1024;
const SAMPLE_RATE: u32 = FRAMERATE * SAMPLES_PER_FRAME;

fn main() {
    let sound_mode = match std::env::args().nth(1) {
        Some(s) => s
            .parse()
            .expect("could not parse first argument (sound mode)"),
        None => SoundMode::default(),
    };
    let tones = tone_stream::Reader::new(std::io::stdin().lock()).unwrap();
    let mut tones: Vec<_> = tones.collect::<Result<_, _>>().unwrap();
    tones.sort_by_key(|t| t.frame);

    let mut audio = Audio::new(SAMPLE_RATE);

    let mut output = std::io::Cursor::new(Vec::new());
    let mut wav = hound::WavWriter::new(
        &mut output,
        hound::WavSpec {
            channels: sound_mode.channels(),
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();

    let mut frame = 0;
    match sound_mode {
        SoundMode::Stereo => {
            for t in tones {
                while frame != t.frame {
                    for _ in 0..SAMPLES_PER_FRAME {
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
                for _ in 0..SAMPLES_PER_FRAME {
                    let audio::StereoSample { left, right } = audio.sample();
                    audio.step_sample();
                    wav.write_sample(left).unwrap();
                    wav.write_sample(right).unwrap();
                }
            }
        }
        SoundMode::Mono => {
            for t in tones {
                while frame != t.frame {
                    for _ in 0..SAMPLES_PER_FRAME {
                        let sample = audio.sample_mono();
                        audio.step_sample();
                        wav.write_sample(sample).unwrap();
                    }
                    frame += 1;
                }

                audio.tone(t.frequency, t.duration, t.volume, t.flags);
            }
            while !audio.has_ended() {
                for _ in 0..SAMPLES_PER_FRAME {
                    let sample = audio.sample_mono();
                    audio.step_sample();
                    wav.write_sample(sample).unwrap();
                }
            }
        }
    };

    wav.finalize().unwrap();
    std::io::stdout()
        .lock()
        .write_all(output.get_ref())
        .unwrap();
}

#[derive(Default, Clone, Copy)]
enum SoundMode {
    Mono = 1,
    #[default]
    Stereo,
}

impl SoundMode {
    fn channels(self) -> u16 {
        self as u16
    }
}

impl std::str::FromStr for SoundMode {
    type Err = ParseSoundModeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "mono" => Ok(SoundMode::Mono),
            "stereo" => Ok(SoundMode::Stereo),
            _ => Err(ParseSoundModeError(())),
        }
    }
}

#[derive(Debug)]
struct ParseSoundModeError(());
