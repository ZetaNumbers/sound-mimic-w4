use crate::{utils::Lerp, FRAMERATE};

/// `-1.0..1.0 => -i16::MAX..i16::MAX`
pub fn sample_from_f32_into_i16(x: f32) -> i16 {
    (x.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
}

#[derive(Default, Clone, Copy)]
pub struct StereoSample<T> {
    pub left: T,
    pub right: T,
}

impl StereoSample<f32> {
    /// `-1.0..1.0 => -i16::MAX..i16::MAX`
    pub fn into_i16(self) -> StereoSample<i16> {
        StereoSample {
            left: sample_from_f32_into_i16(self.left),
            right: sample_from_f32_into_i16(self.right),
        }
    }
}

impl<T> std::ops::AddAssign for StereoSample<T>
where
    T: std::ops::AddAssign,
{
    fn add_assign(&mut self, rhs: Self) {
        self.left += rhs.left;
        self.right += rhs.right;
    }
}

impl<T> std::ops::Add for StereoSample<T>
where
    T: std::ops::Add,
{
    type Output = StereoSample<T::Output>;

    fn add(self, rhs: Self) -> Self::Output {
        StereoSample {
            left: self.left + rhs.left,
            right: self.right + rhs.right,
        }
    }
}

impl<T> std::ops::MulAssign<T> for StereoSample<T>
where
    T: std::ops::MulAssign + Clone,
{
    fn mul_assign(&mut self, rhs: T) {
        self.left *= rhs.clone();
        self.right *= rhs;
    }
}

impl<T> std::ops::Mul<T> for StereoSample<T>
where
    T: std::ops::Mul + Clone,
{
    type Output = StereoSample<T::Output>;

    fn mul(self, rhs: T) -> Self::Output {
        StereoSample {
            left: self.left * rhs.clone(),
            right: self.right * rhs,
        }
    }
}

#[derive(Clone)]
pub struct Audio {
    pub pulse1: Channel,
    pub pulse2: Channel,
    pub triangle: Channel,
    pub noise: Channel,
}

impl Audio {
    pub fn new(sample_rate: u32) -> Self {
        Audio {
            pulse1: Channel::new(sample_rate, 0, 0, 0, 0),
            pulse2: Channel::new(sample_rate, 0, 0, 0, 0),
            triangle: Channel::new(sample_rate, 0, 0, 0, 0),
            noise: Channel::new(sample_rate, 0, 0, 0, 0),
        }
    }

    pub fn sample(&self) -> StereoSample<i16> {
        let mut samples = StereoSample::default();
        samples += self.pulse1.sample_pulse();
        samples += self.pulse2.sample_pulse();
        samples += self.triangle.sample_triangle();
        samples += self.noise.sample_noise();
        samples *= 0.25;
        samples.into_i16()
    }

    pub fn sample_mono(&self) -> i16 {
        let mut sample = 0.0;
        sample += self.pulse1.sample_pulse_mono();
        sample += self.pulse2.sample_pulse_mono();
        sample += self.triangle.sample_triangle_mono();
        sample += self.noise.sample_noise_mono();
        sample *= 0.25;
        sample_from_f32_into_i16(sample)
    }

    pub fn tone(&mut self, frequency: u32, duration: u32, volume: u32, flags: u32) {
        let (sample_rate, phase, channel) = match flags & 0b11 {
            0 => (self.pulse1.sample_rate, self.pulse1.phase, &mut self.pulse1),
            1 => (self.pulse2.sample_rate, self.pulse2.phase, &mut self.pulse2),
            2 => (
                self.triangle.sample_rate,
                self.triangle.phase,
                &mut self.triangle,
            ),
            3 => (self.noise.sample_rate, self.noise.phase, &mut self.noise),
            _ => unreachable!(),
        };
        *channel = Channel::with_phase(sample_rate, frequency, duration, volume, flags, phase);
    }

    pub fn has_ended(&self) -> bool {
        self.pulse1.has_ended()
            && self.pulse2.has_ended()
            && self.triangle.has_ended()
            && self.noise.has_ended()
    }

    pub fn step_sample(&mut self) {
        self.pulse1.step_sample();
        self.pulse2.step_sample();
        self.triangle.step_sample();
        self.noise.step_sample();
    }
}

pub const PULSE1_CHANNEL: u32 = 0;
pub const PULSE2_CHANNEL: u32 = 1;
pub const TRIANGLE_CHANNEL: u32 = 2;
pub const NOISE_CHANNEL: u32 = 3;

#[derive(Clone, Debug)]
pub struct Channel {
    sample_rate: u32,
    start_frequency: f32,
    end_frequency: f32,
    start_at: i64,
    attack_to: i64,
    decay_to: i64,
    sustain_to: i64,
    release_to: i64,
    peak_volume: f32,
    sustain_volume: f32,
    /// measured in 1/8
    duty_cycle: u32,
    pan_left: bool,
    pan_right: bool,
    pub phase: f32,
}

impl Channel {
    pub fn new(sample_rate: u32, frequency: u32, duration: u32, volume: u32, flags: u32) -> Self {
        Channel::with_phase(sample_rate, frequency, duration, volume, flags, 0.0)
    }

    pub fn with_phase(
        sample_rate: u32,
        frequency: u32,
        duration: u32,
        volume: u32,
        flags: u32,
        phase: f32,
    ) -> Self {
        assert_eq!(sample_rate % FRAMERATE, 0);
        let samples_per_frame = sample_rate / FRAMERATE;

        let start_frequency = (frequency & 0xFFFF) as f32;
        let end_frequency = frequency >> 16 & 0xFFFF;
        let end_frequency = if end_frequency != 0 {
            end_frequency as f32
        } else {
            start_frequency
        };

        let durations = Durations::from(duration);
        let attack_to = i64::from(durations.attack_to * samples_per_frame);
        let decay_to = i64::from(durations.decay_to * samples_per_frame);
        let sustain_to = i64::from(durations.sustain_to * samples_per_frame);
        let release_to = i64::from(durations.release_to * samples_per_frame);

        let sustain_volume = (volume & 0xFFFF).min(100) as f32 / 100.0;
        let peak_volume = (volume >> 16 & 0xFFFF).min(100);
        let peak_volume = if peak_volume != 0 {
            peak_volume as f32 / 100.0
        } else {
            sustain_volume
        };

        let mode = flags >> 2 & 0b11;
        let duty_cycle = [1, 2, 4, 6][mode as usize];
        let [pan_left, pan_right] = match flags >> 4 & 0b11 {
            1 => [true, false],
            2 => [false, true],
            _ => [true, true],
        };

        Channel {
            sample_rate,
            start_frequency,
            end_frequency,
            start_at: 0,
            attack_to,
            decay_to,
            sustain_to,
            release_to,
            peak_volume,
            sustain_volume,
            pan_left,
            pan_right,
            duty_cycle,
            phase,
        }
    }

    pub fn has_ended(&self) -> bool {
        self.attack_to <= 0 && self.decay_to <= 0 && self.sustain_to <= 0 && self.release_to <= 0
    }

    pub fn step_sample(&mut self) {
        let af = self.frequency();
        self.start_at = self.start_at.saturating_sub(1);
        self.attack_to = self.attack_to.saturating_sub(1);
        self.decay_to = self.decay_to.saturating_sub(1);
        self.sustain_to = self.sustain_to.saturating_sub(1);
        self.release_to = self.release_to.saturating_sub(1);
        let bf = self.frequency();
        self.phase = (self.phase + 1.0 / self.sample_rate as f32 * 0.5 * (af + bf)).fract();
    }

    pub fn step_frame(&mut self) {
        let af = self.frequency();
        self.start_at = self
            .start_at
            .saturating_sub((self.sample_rate / FRAMERATE).into());
        self.attack_to = self
            .attack_to
            .saturating_sub((self.sample_rate / FRAMERATE).into());
        self.decay_to = self
            .decay_to
            .saturating_sub((self.sample_rate / FRAMERATE).into());
        self.sustain_to = self
            .sustain_to
            .saturating_sub((self.sample_rate / FRAMERATE).into());
        self.release_to = self
            .release_to
            .saturating_sub((self.sample_rate / FRAMERATE).into());
        let bf = self.frequency();
        self.phase =
            (self.phase + FRAMERATE as f32 / self.sample_rate as f32 * 0.5 * (af + bf)).fract();
    }

    pub fn volume(&self) -> f32 {
        if (self.start_at..self.release_to).contains(&0) {
            if (self.start_at..self.attack_to).contains(&0) {
                0.0.unlerp(self.start_at as f32, self.attack_to as f32)
                    .lerp(0.0, self.peak_volume)
            } else if (self.attack_to..self.decay_to).contains(&0) {
                0.0.unlerp(self.attack_to as f32, self.decay_to as f32)
                    .lerp(self.peak_volume, self.sustain_volume)
            } else if (self.decay_to..self.sustain_to).contains(&0) {
                self.sustain_volume
            } else if (self.sustain_to..self.release_to).contains(&0) {
                0.0.unlerp(self.sustain_to as f32, self.release_to as f32)
                    .lerp(self.sustain_volume, 0.0)
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    pub fn frequency(&self) -> f32 {
        if (self.start_at..self.release_to).contains(&0) {
            0.0.unlerp(self.start_at as f32, self.release_to as f32)
                .lerp(self.start_frequency, self.end_frequency)
        } else {
            0.0
        }
    }

    pub fn sample_pulse_mono(&self) -> f32 {
        self.volume() * (self.duty_cycle as f32 - 8.0 * self.phase).signum()
    }

    pub fn sample_triangle_mono(&self) -> f32 {
        self.volume() * (2.0 * (2.0 * self.phase - 1.0).abs() - 1.0)
    }

    pub fn sample_noise_mono(&self) -> f32 {
        0.0
    }

    pub fn sample_pulse(&self) -> StereoSample<f32> {
        self.stereo_norm(self.sample_pulse_mono())
    }

    pub fn sample_triangle(&self) -> StereoSample<f32> {
        self.stereo_norm(self.sample_triangle_mono())
    }

    pub fn sample_noise(&self) -> StereoSample<f32> {
        self.stereo_norm(self.sample_noise_mono())
    }

    fn stereo_norm(&self, sample: f32) -> StereoSample<f32> {
        StereoSample {
            left: if self.pan_left { sample } else { 0.0 },
            right: if self.pan_right { sample } else { 0.0 },
        }
    }
}

#[derive(Clone, Copy)]
pub struct Durations {
    attack_to: u32,
    decay_to: u32,
    sustain_to: u32,
    release_to: u32,
}

impl Durations {
    #[must_use]
    pub fn crop(self, frames: u32) -> Durations {
        Durations {
            attack_to: self.attack_to.min(frames),
            decay_to: self.decay_to.min(frames),
            sustain_to: self.sustain_to.min(frames),
            release_to: self.release_to.min(frames),
        }
    }
}

impl From<u32> for Durations {
    fn from(duration: u32) -> Durations {
        let attack_to = duration >> 24 & 0xFF;
        let decay_to = attack_to + (duration >> 16 & 0xFF);
        let sustain_to = decay_to + (duration & 0xFF);
        let release_to = sustain_to + (duration >> 8 & 0xFF);
        Durations {
            attack_to,
            decay_to,
            sustain_to,
            release_to,
        }
    }
}

impl From<Durations> for u32 {
    fn from(durations: Durations) -> u32 {
        let attack = durations.attack_to;
        assert_eq!(attack & !0xFF, 0);
        let decay = durations.decay_to - durations.attack_to;
        assert_eq!(decay & !0xFF, 0);
        let sustain = durations.sustain_to - durations.decay_to;
        assert_eq!(sustain & !0xFF, 0);
        let release = durations.release_to - durations.sustain_to;
        assert_eq!(release & !0xFF, 0);
        attack << 24 | decay << 16 | sustain | release << 8
    }
}
