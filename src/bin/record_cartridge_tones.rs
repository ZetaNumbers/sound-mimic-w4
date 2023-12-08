use std::{io::Read, path::PathBuf};

use sound_mimic::{cartridge, tone_stream, Cartridge};

#[derive(argh::FromArgs)]
/// Record cartridge played tones
struct RecordCartridgeTones {
    /// path to WASM-4 cartridge
    #[argh(positional)]
    cartridge: PathBuf,
    // TODO: crop tones at the end
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
