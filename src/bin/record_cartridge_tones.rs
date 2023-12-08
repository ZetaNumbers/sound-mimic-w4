use std::{io::Read, path::PathBuf};

use sound_mimic::{audio, cartridge, tone_stream, Cartridge};

#[derive(argh::FromArgs)]
/// Record tones played on a cartridge. Outputs CSV table with recorded tones
/// (`tone` function's arguments) into stdout. Note that tone durations are
/// cropped by default (see `--no-duration-crop`).
struct RecordCartridgeTones {
    /// path to WASM-4 cartridge
    #[argh(positional)]
    cartridge: PathBuf,
    /// amount of frames to record
    #[argh(positional)]
    frames: usize,
    /// load gamestate (memory) from stdin, see
    /// https://github.com/aduros/wasm4/issues/553#issuecomment-1847569775
    /// to learn how to extract gamestate from a running cartridge
    #[argh(switch)]
    load_stdin: bool,
    /// do not crop durations that would last after `frames` have passed, causes
    /// last tones to continue to play even after `frames` have passed, in other
    /// words output unmodified tone arguments
    #[argh(switch)]
    no_duration_crop: bool,
}

fn main() {
    let args: RecordCartridgeTones = argh::from_env();
    let engine = ToneRecorder {
        duration_crop: !args.no_duration_crop,
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
    duration_crop: bool,
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
        if let Some(ends_within) = self
            .ends_within
            .try_into()
            .ok()
            .filter(|_| self.duration_crop)
        {
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
