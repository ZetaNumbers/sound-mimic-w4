use crate::utils::Lerp;

pub const SAMPLE_RATE: u32 = 61440;

#[derive(Default, Clone, Copy)]
pub struct StereoSample<T> {
    pub left: T,
    pub right: T,
}

impl StereoSample<f32> {
    /// `-1.0..1.0 => -i16::MAX..i16::MAX`
    pub fn into_i16(self) -> StereoSample<i16> {
        StereoSample {
            left: (self.left.clamp(-1.0, 1.0) * i16::MAX as f32) as i16,
            right: (self.right.clamp(-1.0, 1.0) * i16::MAX as f32) as i16,
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

#[derive(Default, Clone)]
pub struct Audio {
    pub pulse1: Channel,
    pub pulse2: Channel,
    pub triangle: Channel,
    pub noise: Channel,
}

impl Audio {
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
        sample as i16
    }

    pub fn tone(&mut self, frequency: u32, duration: u32, volume: u32, flags: u32) {
        let (phase, channel) = match flags & 0b11 {
            0 => (self.pulse1.phase, &mut self.pulse1),
            1 => (self.pulse2.phase, &mut self.pulse2),
            2 => (self.triangle.phase, &mut self.triangle),
            3 => (self.noise.phase, &mut self.noise),
            _ => unreachable!(),
        };
        *channel = Channel::with_phase(frequency, duration, volume, flags, phase);
    }

    pub fn step_sample(&mut self) {
        self.pulse1.step_sample();
        self.pulse2.step_sample();
        self.triangle.step_sample();
        self.noise.step_sample();
    }
}

#[derive(Default, Clone, Debug)]
pub struct Channel {
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
    phase: f32,
}

impl Channel {
    pub fn new(frequency: u32, duration: u32, volume: u32, flags: u32) -> Self {
        Channel::with_phase(frequency, duration, volume, flags, 0.0)
    }

    pub fn with_phase(frequency: u32, duration: u32, volume: u32, flags: u32, phase: f32) -> Self {
        let start_frequency = (frequency & 0xFFFF) as f32;
        let end_frequency = frequency >> 16 & 0xFFFF;
        let end_frequency = if end_frequency != 0 {
            end_frequency as f32
        } else {
            start_frequency
        };

        let attack_to = i64::from((duration >> 24 & 0xFF) * SAMPLE_RATE);
        let decay_to = attack_to + i64::from((duration >> 16 & 0xFF) * SAMPLE_RATE);
        let sustain_to = decay_to + i64::from((duration & 0xFF) * SAMPLE_RATE);
        let release_to = sustain_to + i64::from((duration >> 8 & 0xFF) * SAMPLE_RATE);

        let sustain_volume = (volume & 0xFFFF).min(100) as f32 / 100.0;
        let peak_volume = (volume >> 16 & 0xFFFF).min(100);
        let peak_volume =
            if peak_volume != 0 {
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

    pub fn step_sample(&mut self) {
        let af = self.frequency();
        self.start_at = self.start_at.saturating_sub(1);
        self.attack_to = self.attack_to.saturating_sub(1);
        self.decay_to = self.decay_to.saturating_sub(1);
        self.sustain_to = self.sustain_to.saturating_sub(1);
        self.release_to = self.release_to.saturating_sub(1);
        let bf = self.frequency();
        self.phase = (self.phase + 1.0 / SAMPLE_RATE as f32 * 0.5 * (af + bf)).fract();
    }

    pub fn step_frame(&mut self) {
        let af = self.frequency();
        self.start_at = self.start_at.saturating_sub((SAMPLE_RATE / 60).into());
        self.attack_to = self.attack_to.saturating_sub((SAMPLE_RATE / 60).into());
        self.decay_to = self.decay_to.saturating_sub((SAMPLE_RATE / 60).into());
        self.sustain_to = self.sustain_to.saturating_sub((SAMPLE_RATE / 60).into());
        self.release_to = self.release_to.saturating_sub((SAMPLE_RATE / 60).into());
        let bf = self.frequency();
        self.phase = (self.phase + 60.0 / SAMPLE_RATE as f32 * 0.5 * (af + bf)).fract();
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

    fn sample_pulse_mono(&self) -> f32 {
        self.volume() * (self.duty_cycle as f32 - 8.0 * self.phase).signum()
    }

    fn sample_triangle_mono(&self) -> f32 {
        self.volume() * (2.0 * (2.0 * self.phase - 1.0).abs() - 1.0)
    }

    fn sample_noise_mono(&self) -> f32 {
        0.0
    }

    fn sample_pulse(&self) -> StereoSample<f32> {
        self.stereo_norm(self.sample_pulse_mono())
    }

    fn sample_triangle(&self) -> StereoSample<f32> {
        self.stereo_norm(self.sample_triangle_mono())
    }

    fn sample_noise(&self) -> StereoSample<f32> {
        self.stereo_norm(self.sample_noise_mono())
    }

    fn stereo_norm(&self, sample: f32) -> StereoSample<f32> {
        StereoSample {
            left: if self.pan_left { sample } else { 0.0 },
            right: if self.pan_right { sample } else { 0.0 },
        }
    }
}
