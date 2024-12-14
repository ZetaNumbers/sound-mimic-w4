pub use apu::Apu;
pub const FRAMERATE: u32 = 60;

pub mod apu;
pub mod tone_stream;

// TODO: consider transfering executable interface into library API
