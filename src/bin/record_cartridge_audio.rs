use std::env;

use sound_mimic::{audio, Cartridge};

fn main() {
    let cartridge = env::args_os()
        .nth(1)
        .expect("No first argument (cartridge) was provided");

    let out_recording = env::args_os()
        .nth(2)
        .expect("No second argument (out recording) was provided");

    let frames: u32 = env::args()
        .nth(3)
        .expect("No third argument (frames) was provided")
        .parse()
        .expect("Could not properly parse first argument");

    let mut cart = Cartridge::load(cartridge);

    let mut wav = hound::WavWriter::create(
        out_recording,
        hound::WavSpec {
            channels: 2,
            sample_rate: audio::SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();

    for _ in 0..frames {
        cart.update();
        let audio = cart.audio_mut();
        for _ in 0..audio::SAMPLE_RATE / 60 {
            let audio::StereoSample { left, right } = audio.sample();
            audio.step_sample();
            wav.write_sample(left).unwrap();
            wav.write_sample(right).unwrap();
        }
    }

    wav.finalize().unwrap();
}
