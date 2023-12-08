//! Stream tones across applications

use std::io;

pub const HEADER: &str = "frame,frequency,duration,volume,flags";

pub struct ToneRecord {
    pub frame: u64,
    pub frequency: u32,
    pub duration: u32,
    pub volume: u32,
    pub flags: u32,
}

impl ToneRecord {
    fn parse_line(line: &str, line_number: u64) -> Result<Self, ReaderError> {
        let mut values = line.split(',');
        let frame = values
            .next()
            .ok_or_else(|| ReaderError::not_enougth_values(line_number, "frame"))?;
        let frequency = values
            .next()
            .ok_or_else(|| ReaderError::not_enougth_values(line_number, "frequency"))?;
        let duration = values
            .next()
            .ok_or_else(|| ReaderError::not_enougth_values(line_number, "duration"))?;
        let volume = values
            .next()
            .ok_or_else(|| ReaderError::not_enougth_values(line_number, "volume"))?;
        let flags = values
            .next()
            .ok_or_else(|| ReaderError::not_enougth_values(line_number, "flags"))?;

        if values.next().is_some() {
            return Err(ReaderError::too_many_values(line_number));
        }

        Ok(ToneRecord {
            frame: frame
                .parse()
                .map_err(|e| ReaderError::from_parse_error(e, line_number, "frame"))?,
            frequency: frequency
                .parse()
                .map_err(|e| ReaderError::from_parse_error(e, line_number, "frequency"))?,
            duration: duration
                .parse()
                .map_err(|e| ReaderError::from_parse_error(e, line_number, "duration"))?,
            volume: volume
                .parse()
                .map_err(|e| ReaderError::from_parse_error(e, line_number, "volume"))?,
            flags: flags
                .parse()
                .map_err(|e| ReaderError::from_parse_error(e, line_number, "flags"))?,
        })
    }
}

pub struct Reader<R> {
    line_number: u64,
    lines: io::Lines<R>,
}

impl<R: io::BufRead> Reader<R> {
    pub fn new(reader: R) -> Result<Self, ReaderError> {
        let mut lines = io::BufRead::lines(reader);
        let header = lines
            .next()
            .unwrap_or_else(|| Err(io::ErrorKind::UnexpectedEof.into()))
            .map_err(|e| ReaderError::from_io_error(e, 0))?;
        if header != HEADER {
            return Err(ReaderError::invalid_header());
        }
        Ok(Reader {
            lines,
            line_number: 1,
        })
    }
}

impl<R: io::BufRead> Iterator for Reader<R> {
    type Item = Result<ToneRecord, ReaderError>;

    fn next(&mut self) -> Option<Self::Item> {
        let line = self.lines.next()?;
        let line_number = self.line_number;
        self.line_number = line_number.checked_add(1).unwrap();
        let line = match line {
            Ok(l) => l,
            Err(io_error) => return Some(Err(ReaderError::from_io_error(io_error, line_number))),
        };
        Some(ToneRecord::parse_line(&line, line_number))
    }
}

pub struct Writer<W> {
    frame: u64,
    writer: W,
}

impl<W: io::Write> Writer<W> {
    pub fn new(mut writer: W) -> io::Result<Self> {
        writeln!(writer, "{HEADER}")?;
        Ok(Writer { frame: 0, writer })
    }

    pub fn write_tone(
        &mut self,
        frequency: u32,
        duration: u32,
        volume: u32,
        flags: u32,
    ) -> io::Result<()> {
        let frame = self.frame;
        writeln!(
            self.writer,
            "{frame},{frequency},{duration},{volume},{flags}"
        )
    }
}

#[derive(Debug)]
pub struct ReaderError {
    line_number: u64,
    field_name: Option<&'static str>,
    io_error: io::Error,
}

impl ReaderError {
    pub fn line_number(&self) -> u64 {
        self.line_number
    }

    pub fn field_name(&self) -> Option<&'static str> {
        self.field_name
    }

    fn from_io_error(io_error: io::Error, line_number: u64) -> Self {
        ReaderError {
            io_error,
            line_number,
            field_name: None,
        }
    }

    fn not_enougth_values(line_number: u64, field_name: &'static str) -> Self {
        ReaderError {
            io_error: io::Error::new(
                io::ErrorKind::InvalidInput,
                ReaderErrorKind::NotEnoughValues,
            ),
            line_number,
            field_name: Some(field_name),
        }
    }

    fn too_many_values(line_number: u64) -> Self {
        ReaderError {
            io_error: io::Error::new(io::ErrorKind::InvalidInput, ReaderErrorKind::TooManyValues),
            line_number,
            field_name: None,
        }
    }

    fn invalid_header() -> Self {
        ReaderError {
            io_error: io::Error::new(io::ErrorKind::InvalidInput, ReaderErrorKind::InvalidHeader),
            line_number: 0,
            field_name: None,
        }
    }

    fn from_parse_error(
        error: std::num::ParseIntError,
        line_number: u64,
        field_name: &'static str,
    ) -> Self {
        ReaderError {
            io_error: io::Error::new(io::ErrorKind::InvalidInput, error),
            line_number,
            field_name: Some(field_name),
        }
    }
}

impl std::fmt::Display for ReaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} on line {}", self.io_error, self.line_number)?;
        if let Some(field_name) = self.field_name {
            write!(f, " field `{field_name}`")?;
        }
        Ok(())
    }
}

impl std::error::Error for ReaderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.io_error)
    }
}

#[non_exhaustive]
#[derive(Debug)]
pub enum ReaderErrorKind {
    InvalidHeader,
    NotEnoughValues,
    TooManyValues,
    ParseError(std::num::ParseIntError),
}

impl std::fmt::Display for ReaderErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReaderErrorKind::InvalidHeader => "invalid CSV header".fmt(f),
            ReaderErrorKind::NotEnoughValues => "not enough CSV field values".fmt(f),
            ReaderErrorKind::TooManyValues => "too many CSV field values".fmt(f),
            ReaderErrorKind::ParseError(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for ReaderErrorKind {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ReaderErrorKind::ParseError(e) => Some(e),
            _ => None,
        }
    }
}

impl<W> Writer<W> {
    pub fn step_frame(&mut self) -> Result<(), FrameCountOverflowError> {
        self.frame = self
            .frame
            .checked_add(1)
            .ok_or(FrameCountOverflowError(()))?;
        Ok(())
    }
}
#[derive(Debug)]
pub struct FrameCountOverflowError(());

impl std::fmt::Display for FrameCountOverflowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        "frame counter overflowed".fmt(f)
    }
}

impl std::error::Error for FrameCountOverflowError {}
