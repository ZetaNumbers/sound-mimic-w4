use std::{
    env,
    io::{Read, Write},
};

use sound_mimic::{audio, Cartridge};

fn main() {
    let mut cartridge = Vec::new();
    std::io::stdin().lock().read_to_end(&mut cartridge).unwrap();
    let mut cartridge = Cartridge::new(cartridge);

    let frames: u32 = env::args()
        .nth(1)
        .expect("No first argument (frames) was provided")
        .parse()
        .expect("Could not properly parse first argument");

    let mut output = std::io::Cursor::new(Vec::new());
    let mut wav = hound::WavWriter::new(
        &mut output,
        hound::WavSpec {
            channels: 2,
            sample_rate: audio::SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();

    for _ in 0..frames {
        cartridge.update();
        let audio = cartridge.audio_mut();
        for _ in 0..audio::SAMPLE_RATE / 60 {
            let audio::StereoSample { left, right } = audio.sample();
            audio.step_sample();
            wav.write_sample(left).unwrap();
            wav.write_sample(right).unwrap();
        }
    }

    wav.finalize().unwrap();
    std::io::stdout()
        .lock()
        .write_all(output.get_ref())
        .unwrap();
}
