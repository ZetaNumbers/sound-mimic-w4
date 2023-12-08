use std::{io::Read, path::PathBuf};

use sound_mimic::{audio, cartridge, tone_stream, Cartridge};

#[derive(argh::FromArgs)]
/// Record cartridge played tones
struct RecordCartridgeTones {
    /// path to WASM-4 cartridge
    #[argh(positional)]
    cartridge: PathBuf,
    /// amount of frames to record
    #[argh(positional)]
    frames: usize,
    /// load gamestate (memory) from stdin
    #[argh(switch)]
    load_stdin: bool,
}

fn main() {
    let args: RecordCartridgeTones = argh::from_env();
    let engine = ToneRecorder {
        ends_within: args.frames,
        writer: tone_stream::Writer::new(std::io::stdout().lock()).unwrap(),
    };
    let load_memory = args.load_stdin.then(|| {
        let mut buf = Box::new([0; cartridge::MEMORY_SIZE]);
        std::io::stdin().lock().read_exact(&mut *buf).unwrap();
        buf
    });
    let mut cartridge = Cartridge::new(args.cartridge, engine, load_memory.as_deref()).unwrap();

    for _ in 0..args.frames {
        cartridge.update().unwrap();
    }
}

struct ToneRecorder<W> {
    ends_within: usize,
    writer: tone_stream::Writer<W>,
}

impl<W: std::io::Write> cartridge::Engine for ToneRecorder<W> {
    fn tone(
        &mut self,
        frequency: u32,
        mut duration: u32,
        volume: u32,
        flags: u32,
    ) -> anyhow::Result<()> {
        if let Ok(ends_within) = self.ends_within.try_into() {
            duration = audio::Durations::from(duration).crop(ends_within).into();
        }

        self.writer.write_tone(frequency, duration, volume, flags)?;
        Ok(())
    }

    fn after_update(&mut self) -> anyhow::Result<()> {
        self.ends_within = self.ends_within.saturating_sub(1);
        self.writer.step_frame()?;
        Ok(())
    }
}
