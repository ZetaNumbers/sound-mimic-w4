use std::{env, io::Read};

use sound_mimic::{cartridge, tone_stream, Cartridge};

fn main() {
    let mut cartridge = Vec::new();
    std::io::stdin().lock().read_to_end(&mut cartridge).unwrap();
    let engine = ToneRecorder {
        writer: tone_stream::Writer::new(std::io::stdout().lock()).unwrap(),
    };
    let mut cartridge = Cartridge::new(cartridge, engine).unwrap();

    let frames: u32 = env::args()
        .nth(1)
        .expect("No first argument (frames) was provided")
        .parse()
        .expect("Could not properly parse first argument");

    for _ in 0..frames {
        cartridge.update().unwrap();
    }
}

struct ToneRecorder<W> {
    writer: tone_stream::Writer<W>,
}

impl<W: std::io::Write> cartridge::Engine for ToneRecorder<W> {
    fn tone(
        &mut self,
        frequency: u32,
        duration: u32,
        volume: u32,
        flags: u32,
    ) -> anyhow::Result<()> {
        self.writer.write_tone(frequency, duration, volume, flags)?;
        Ok(())
    }

    fn after_update(&mut self) -> anyhow::Result<()> {
        self.writer.step_frame()?;
        Ok(())
    }
}
