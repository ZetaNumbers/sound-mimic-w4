# Sound mimic for WASM-4

This project is a set of tools to manipulate and analyze sounds for the purpose to make better sounds for WASM-4 cartridges.

## Project layout and usage

Sound mimic contains a single cargo package with multiple binaries. Each binary usualy structured around its stdin and stdout, please check out `--help` to familiarize yourself with the tools. There's also gitignored assets directoery for your experiments.

<!-- WARN: Please update cli documentation if documentation below is changed -->
- `record_cartridge_tones` - Record tones played on a cartridge. Outputs CSV table with recorded tones (`tone` function's arguments) into stdout. Note that tone durations are cropped by default (see `--no-duration-crop`).
- `tone_playback` - Transforms tone CSV table from stdin into WAV file and outputs it into stdout.
- `mimic` - Chooses most suitable tones for each frame of input sound (wav file) and outputs these tones as CSV table in stdout.
- `spectrogram` - Render spectogram of a sound (WAV file) from stdin into image and output it into stdout. Spectogram highlights sound intensity level, not amplitude.

While tone CSV tables are most straight forward, to interact with output media you have to save it with file and somehow open it, or pipe it into right program.

- For audio WAV files I use `vlc --play-and-exit --intf dummy` command to playback them. Obviously make sure [VLC](https://www.videolan.org/vlc/) is installed and is accessible from your shell. Pass `-` instead of WAV file path to load data from stdin instead.
- For image files I use exclusive to wezterm `wezterm imgcat` command. It renders images right within wezterm terminal. It can also get image from stdin. I prefer using [QOI image format](https://en.wikipedia.org/wiki/QOI_(image_format)) for it's light-weight encoding. If you aren't using wezterm consider saving image onto disk and then opening it like you would any other picture and learn how to do it from the terminal.

### Examples

```bash
# look at cartridge.wasm tones for 1 second (60 frames) of execution
cargo run --bin record_cartridge_tones -- assets/cartridge.wasm 60

# save cartridge.wasm tones for 10 seconds (600 frames) of execution
cargo run --bin record_cartridge_tones -- assets/cartridge.wasm 600 | cargo run --bin tone_playback > assets/cartridge_recording.wav

# save spectrogram of a sound.wav into sound_spectrogram.png
cat assets/sound.wav | cargo run --bin spectrogram -- png > assets/sound_spectrogram.png

# generate and playback mimic tones of a sound.wav (use `-r` flag to run in release mode)
cat assets/sound.wav | cargo run --bin mimic -r | cargo run --bin tone_playback | vlc --play-and-exit --intf dummy -
```
