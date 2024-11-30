pub use crate::{apu::Apu, audio::Audio, cartridge::Cartridge};

pub const FRAMERATE: u32 = 60;

mod apu;
pub mod audio;
pub mod cartridge;
pub mod tone_stream;
pub mod utils;

// TODO: consider transfering executable interface into library API
