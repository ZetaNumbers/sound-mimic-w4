pub use crate::{audio::Audio, cartridge::Cartridge};

pub const FRAMERATE: u32 = 60;

pub mod apu;
pub mod audio;
pub mod cartridge;
pub mod tone_stream;
pub mod utils;

// TODO: consider transfering executable interface into library API
