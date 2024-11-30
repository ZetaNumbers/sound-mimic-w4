pub use crate::{apu::Apu, cartridge::Cartridge};

pub const FRAMERATE: u32 = 60;

pub mod apu;
pub mod cartridge;
pub mod tone_stream;
pub mod utils;

// TODO: consider transfering executable interface into library API
