use std::{cmp::min, iter};

#[derive(Clone, Copy)]
struct NoiseAddend {
    seed: u16,
    last_random: i16,
}

impl Default for NoiseAddend {
    fn default() -> Self {
        NoiseAddend {
            seed: 0x1,
            last_random: 0,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct PulseAddend {
    duty_cycle: f32,
}

enum ChannelAddendMut<'a> {
    Pulse(&'a mut PulseAddend),
    Triangle,
    Noise(&'a mut NoiseAddend),
}

#[derive(Default)]
struct GeneralChannel {
    freq1: f32,
    freq2: f32,
    start_time: u64,
    attack_time: u64,
    decay_time: u64,
    sustain_time: u64,
    release_time: u64,
    end_tick: u64,
    sustain_volume: i16,
    peak_volume: i16,
    phase: f32,
    pan: u8,
}

#[derive(Default)]
struct Channel<A> {
    general: GeneralChannel,
    addend: A,
}

#[derive(Default)]
struct Channels {
    pulse: [Channel<PulseAddend>; 2],
    triangle: Channel<()>,
    noise: Channel<NoiseAddend>,
}

impl Channels {
    fn get_mut(&mut self, idx: usize) -> Option<(&mut GeneralChannel, ChannelAddendMut<'_>)> {
        match idx {
            0 | 1 => {
                let c = &mut self.pulse[idx];
                Some((&mut c.general, ChannelAddendMut::Pulse(&mut c.addend)))
            }
            2 => Some((&mut self.triangle.general, ChannelAddendMut::Triangle)),
            3 => Some((
                &mut self.noise.general,
                ChannelAddendMut::Noise(&mut self.noise.addend),
            )),
            _ => None,
        }
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = (&mut GeneralChannel, ChannelAddendMut<'_>)> {
        self.pulse
            .iter_mut()
            .map(|c| (&mut c.general, ChannelAddendMut::Pulse(&mut c.addend)))
            .chain(iter::once((
                &mut self.triangle.general,
                ChannelAddendMut::Triangle,
            )))
            .chain(iter::once((
                &mut self.noise.general,
                ChannelAddendMut::Noise(&mut self.noise.addend),
            )))
    }
}

#[derive(Default)]
pub struct Apu {
    channels: Channels,
    time: u64,
    ticks: u64,
}

fn lerp(value1: i32, value2: i32, t: f32) -> i32 {
    (value1 as f32 + t * (value2 - value1) as f32) as i32
}

fn lerpf(value1: f32, value2: f32, t: f32) -> f32 {
    value1 + t * (value2 - value1)
}

fn ramp(time: u64, value1: i32, value2: i32, time1: u64, time2: u64) -> i32 {
    if time >= time2 {
        return value2;
    }
    let t: f32 = time.wrapping_sub(time1) as f32 / time2.wrapping_sub(time1) as f32;
    lerp(value1, value2, t)
}

fn rampf(time: u64, value1: f32, value2: f32, time1: u64, time2: u64) -> f32 {
    if time >= time2 {
        return value2;
    }
    let t: f32 = time.wrapping_sub(time1) as f32 / time2.wrapping_sub(time1) as f32;
    lerpf(value1, value2, t)
}

fn polyblep(phase: f32, phase_inc: f32) -> f32 {
    if phase < phase_inc {
        let t: f32 = phase / phase_inc;
        t + t - t * t
    } else if phase > 1.0f32 - phase_inc {
        let t_0: f32 = (phase - (1.0f32 - phase_inc)) / phase_inc;
        1.0f32 - (t_0 + t_0 - t_0 * t_0)
    } else {
        1.0f32
    }
}

fn midi_freq(note: u8, bend: u8) -> f32 {
    2.0f32.powf((note as f32 - 69.0f32 + bend as f32 / 256.0f32) / 12.0f32) * 440.0f32
}

impl Apu {
    pub fn new() -> Apu {
        Apu::default()
    }

    pub fn tick(&mut self) {
        self.ticks = self.ticks.checked_add(1).expect("APU tick overflow");
    }

    pub fn tone(&mut self, frequency: i32, duration: i32, volume: i32, flags: i32) {
        let freq1 = frequency & 0xffff;
        let freq2 = frequency >> 16 & 0xffff;
        let sustain = duration & 0xff;
        let release = duration >> 8 & 0xff;
        let decay = duration >> 16 & 0xff;
        let attack = duration >> 24 & 0xff;
        let sustain_volume = min(volume & 0xff, 100);
        let peak_volume = min(volume >> 8 & 0xff, 100);
        let channel_idx = (flags & 0x3) as usize;
        let mode = flags >> 2 & 0x3;
        let pan = flags >> 4 & 0x3;
        let note_mode = flags & 0x40;
        let (channel, channel_addend) = self.channels.get_mut(channel_idx).unwrap();
        if self.time > channel.release_time && self.ticks != channel.end_tick {
            channel.phase = (if channel_idx == 2 { 0.25f64 } else { 0. }) as f32;
        }
        if note_mode != 0 {
            channel.freq1 = midi_freq(freq1 as u8, (freq1 >> 8) as u8);
            channel.freq2 = if freq2 == 0_i32 {
                0.
            } else {
                midi_freq((freq2 & 0xff_i32) as u8, (freq2 >> 8_i32) as u8)
            };
        } else {
            channel.freq1 = freq1 as f32;
            channel.freq2 = freq2 as f32;
        }
        channel.start_time = self.time;
        channel.attack_time =
            (channel.start_time).wrapping_add((44100_i32 * attack / 60_i32) as u64);
        channel.decay_time =
            (channel.attack_time).wrapping_add((44100_i32 * decay / 60_i32) as u64);
        channel.sustain_time =
            (channel.decay_time).wrapping_add((44100_i32 * sustain / 60_i32) as u64);
        channel.release_time =
            (channel.sustain_time).wrapping_add((44100_i32 * release / 60_i32) as u64);
        channel.end_tick = self
            .ticks
            .wrapping_add(attack as u64)
            .wrapping_add(decay as u64)
            .wrapping_add(sustain as u64)
            .wrapping_add(release as u64);
        let max_volume = if channel_idx == 2 { 0x2000_i16 } else { 0x1333 };
        channel.sustain_volume = (max_volume as i32 * sustain_volume / 100_i32) as i16;
        channel.peak_volume = if peak_volume != 0 {
            (max_volume as i32 * peak_volume / 100_i32) as i16
        } else {
            max_volume
        };
        channel.pan = pan as u8;
        match channel_addend {
            ChannelAddendMut::Pulse(channel_addend) => match mode {
                0 => {
                    channel_addend.duty_cycle = 0.125f32;
                }
                2 => {
                    channel_addend.duty_cycle = 0.5f32;
                }
                1 | 3 => {
                    channel_addend.duty_cycle = 0.25f32;
                }
                _ => unreachable!(),
            },
            ChannelAddendMut::Triangle if release == 0_i32 => {
                channel.release_time =
                    (channel.release_time).wrapping_add((44100_i32 / 1000_i32) as u64);
            }
            _ => (),
        }
    }
}

impl Iterator for Apu {
    type Item = [i16; 2];

    fn next(&mut self) -> Option<[i16; 2]> {
        let new_time = self.time.checked_add(1)?;
        let mut mix_left = 0;
        let mut mix_right = 0;
        for (channel, channel_addend) in self.channels.iter_mut() {
            if self.time < channel.release_time || self.ticks == channel.end_tick {
                let freq = channel.get_current_frequency(self.time);
                let volume = channel.get_current_volume(self.time);
                let sample;
                match channel_addend {
                    ChannelAddendMut::Noise(channel_addend) => {
                        channel.phase += freq * freq / (1000000.0f32 / 44100. * 44100.);
                        while channel.phase > 0. {
                            channel.phase -= 1.;
                            channel_addend.seed = (channel_addend.seed as i32
                                ^ channel_addend.seed as i32 >> 7_i32)
                                as u16;
                            channel_addend.seed = (channel_addend.seed as i32
                                ^ (channel_addend.seed as i32) << 9_i32)
                                as u16;
                            channel_addend.seed = (channel_addend.seed as i32
                                ^ channel_addend.seed as i32 >> 13_i32)
                                as u16;
                            channel_addend.last_random =
                                (2_i32 * (channel_addend.seed as i32 & 0x1_i32) - 1_i32) as i16;
                        }
                        sample = (volume as i32 * channel_addend.last_random as i32) as i16;
                    }
                    _ => {
                        let phase_inc = freq / 44100.;
                        channel.phase += phase_inc;
                        if channel.phase >= 1. {
                            channel.phase -= 1.;
                        }
                        match channel_addend {
                            ChannelAddendMut::Pulse(channel_addend) => {
                                let duty_phase;
                                let duty_phase_inc;
                                let multiplier;
                                if channel.phase < channel_addend.duty_cycle {
                                    duty_phase = channel.phase / channel_addend.duty_cycle;
                                    duty_phase_inc = phase_inc / channel_addend.duty_cycle;
                                    multiplier = volume;
                                } else {
                                    duty_phase = (channel.phase - channel_addend.duty_cycle)
                                        / (1.0f32 - channel_addend.duty_cycle);
                                    duty_phase_inc =
                                        phase_inc / (1.0f32 - channel_addend.duty_cycle);
                                    multiplier = -(volume as i32) as i16;
                                }
                                sample = (multiplier as f32 * polyblep(duty_phase, duty_phase_inc))
                                    as i16;
                            }
                            ChannelAddendMut::Triangle => {
                                sample = (volume as f32
                                    * (2. * (2. * channel.phase - 1.).abs() - 1.))
                                    as i16;
                            }
                            ChannelAddendMut::Noise(_) => unreachable!(),
                        }
                    }
                }
                if channel.pan as i32 != 1_i32 {
                    mix_right = (mix_right as i32 + sample as i32) as i16;
                }
                if channel.pan as i32 != 2_i32 {
                    mix_left = (mix_left as i32 + sample as i32) as i16;
                }
            }
        }
        self.time = new_time;
        Some([mix_left, mix_right])
    }
}

impl GeneralChannel {
    fn get_current_frequency(&self, time: u64) -> f32 {
        if self.freq2 > 0. {
            rampf(
                time,
                self.freq1,
                self.freq2,
                self.start_time,
                self.release_time,
            )
        } else {
            self.freq1
        }
    }

    fn get_current_volume(&self, time: u64) -> i16 {
        if time >= self.sustain_time
            && (self.release_time).wrapping_sub(self.sustain_time) > (44100_i32 / 1000_i32) as u64
        {
            ramp(
                time,
                self.sustain_volume as i32,
                0_i32,
                self.sustain_time,
                self.release_time,
            ) as i16
        } else if time >= self.decay_time {
            self.sustain_volume
        } else if time >= self.attack_time {
            ramp(
                time,
                self.peak_volume as i32,
                self.sustain_volume as i32,
                self.attack_time,
                self.decay_time,
            ) as i16
        } else {
            ramp(
                time,
                0_i32,
                self.peak_volume as i32,
                self.start_time,
                self.attack_time,
            ) as i16
        }
    }
}
