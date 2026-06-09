// # Quick play - uses misaki by default
// synth "Hello world" --play
//
// # Compare phonemizers side by side
// synth "Hello world" -p misaki --play
// synth "Hello world" -p espeak --play
//
// # Different voice
// synth "Hello world" -v am_adam --play
//
// # Save to file (mp3 or wav)
// synth "Hello world" -v af_heart -o test.wav
//
// # List available voices
// synth --list-voices
//
// ## Here are test sentences hitting the patterns most likely to trip up misaki's G2P — letter-by-letter acronyms, mixed acronyms-as-words, possessives, and plurals:
//
// Letter-by-letter acronyms that look like words:
//
//
// The UI for the REST API is broken — the backend returns SQL errors instead of JSON.
// Acronym possessives and plurals:
//
//
// NASA's JWST captured UV and IR data, but the ML team's GPUs were all busy running NLP jobs.
// Dense acronym sentence (high error probability):
//
//
// The CEO asked the CTO to review the SaaS platform's AWS S3 costs, the CDN latency, and the CI/CD pipeline before the IPO.
// Acronym followed by numbers:
//
//
// Enable HTTP/2 on port 443, set the TTL to 3600, and configure OAuth2 before the MVP ships.
// Mixed pronounce-as-word vs spell-out:
//
//
// NATO's use of RADAR and LASER technology predates USB, HTML, and modern CSS frameworks.
// Run them like this to compare side by side:
//
//
// # misaki (default)
// synth "The UI for the REST API is broken — the backend returns SQL errors instead of JSON." --play
//
// # espeak
// synth "The UI for the REST API is broken — the backend returns SQL errors instead of JSON." -p espeak --play

use std::{
    io::Read,
    path::PathBuf,
    process::Command,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use clap::Parser;
use thiserror::Error;
use tts_rs::{
    inference::{TtsEngine, encode_mp3, encode_wav},
    phonemizer::{EspeakBackend, MisakiBackend, Phonemizer},
    voices::VoiceBank,
};

#[derive(Debug, Error)]
enum SynthError {
    #[error("voice bank: {0}")]
    Voice(#[from] tts_rs::error::VoiceError),
    #[error("phonemizer: {0}")]
    Phonemizer(#[from] tts_rs::error::PhonemizerError),
    #[error("model: {0}")]
    Inference(#[from] tts_rs::error::InferenceError),
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("empty input text")]
    EmptyInput,
    #[error("playback failed: {0}")]
    PlaybackFailed(String),
}

#[derive(Parser)]
#[command(
    name = "synth",
    about = "Synthesize speech from text using the local Kokoro TTS model",
    long_about = "Synthesize speech and write MP3/WAV output or play it directly.\n\
                  Useful for quickly comparing phonemizer backends and voice quality.\n\n\
                  EXAMPLES:\n  \
                    synth \"Hello world\" --play\n  \
                    synth \"Hello world\" -p misaki --play\n  \
                    synth \"Hello world\" -v am_adam -o test.wav\n  \
                    echo \"Hello world\" | synth --play"
)]
struct Cli {
    /// Text to synthesize. Pass '-' to read from stdin.
    #[arg(default_value = "-")]
    text: String,

    /// Voice name (e.g. af_heart, am_adam). Use --list-voices to see all.
    #[arg(short, long, default_value = "af_heart")]
    voice: String,

    /// Phonemizer backend. espeak requires libespeak-ng (brew install espeak-ng on macOS).
    #[arg(short, long, value_enum, default_value_t = PhonemizerChoice::Misaki)]
    phonemizer: PhonemizerChoice,

    /// Speed multiplier (0.5 = half speed, 2.0 = double)
    #[arg(short, long, default_value_t = 1.0)]
    speed: f32,

    /// Save output to file. Format is inferred from extension (.mp3 or .wav).
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Play audio immediately via afplay (macOS). Uses a temp file, no permanent save.
    #[arg(short = 'P', long)]
    play: bool,

    /// Directory containing model.onnx, vocab.json, and voices/
    #[arg(short, long)]
    model_dir: Option<PathBuf>,

    /// Print all available voice names and exit
    #[arg(long)]
    list_voices: bool,
}

#[derive(clap::ValueEnum, Clone, Debug)]
enum PhonemizerChoice {
    Misaki,
    Espeak,
}

#[tokio::main]
async fn main() -> Result<(), SynthError> {
    let cli = Cli::parse();

    let model_dir = cli.model_dir.unwrap_or_else(|| {
        PathBuf::from(std::env::var("MODEL_DIR").unwrap_or_else(|_| "./models".into()))
    });

    let voices_dir = model_dir.join("voices");
    let voice_bank = VoiceBank::load(&voices_dir)?;

    if cli.list_voices {
        println!("Available voices:");
        for name in voice_bank.voice_names() {
            println!("  {name}");
        }
        return Ok(());
    }

    if cli.output.is_none() && !cli.play {
        eprintln!("hint: use --output <file> to save or --play to play directly");
    }

    // Resolve input text
    let text = if cli.text == "-" {
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf)?;
        buf.trim().to_string()
    } else {
        cli.text.clone()
    };

    if text.is_empty() {
        return Err(SynthError::EmptyInput);
    }

    // Load phonemizer
    let backend: Box<dyn tts_rs::phonemizer::PhonemizerBackend> = match cli.phonemizer {
        PhonemizerChoice::Misaki => Box::new(MisakiBackend),
        PhonemizerChoice::Espeak => Box::new(EspeakBackend),
    };
    let phonemizer_name = backend.name().to_owned();

    let phonemizer = Phonemizer::load(&model_dir.join("vocab.json"), backend)?;

    let model_path = model_dir.join("model_q4.onnx");
    eprintln!("loading model from {} …", model_path.display());
    let engine = Arc::new(TtsEngine::load(&model_path)?);

    eprintln!(
        "phonemizer: {phonemizer_name}  voice: {}  speed: {}",
        cli.voice, cli.speed
    );
    let token_ids = phonemizer.phonemize(&text)?;
    eprintln!("token_ids: {} tokens", token_ids.len());

    let style = voice_bank.get_style(&cli.voice, token_ids.len())?;

    eprintln!("synthesizing…");
    let samples = engine.synthesize(&token_ids, &style, cli.speed).await?;

    let duration_secs = samples.len() as f32 / 24_000.0;
    eprintln!(
        "generated {:.2}s of audio ({} samples)",
        duration_secs,
        samples.len()
    );

    // Save to output file if requested
    if let Some(ref output_path) = cli.output {
        let ext = output_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("mp3")
            .to_lowercase();

        let bytes = match ext.as_str() {
            "wav" => encode_wav(&samples, 24_000)?,
            _ => encode_mp3(&samples, 24_000)?,
        };
        std::fs::write(output_path, &bytes)?;
        eprintln!("written {} bytes to {}", bytes.len(), output_path.display());
    }

    // Play via afplay
    if cli.play {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let audio_path = std::env::temp_dir().join(format!("synth-{ts}.mp3"));

        let mp3 = encode_mp3(&samples, 24_000)?;
        std::fs::write(&audio_path, &mp3)?;

        eprintln!("playing via afplay…");
        let status = Command::new("afplay")
            .arg(&audio_path)
            .status()
            .map_err(|e| SynthError::PlaybackFailed(format!("afplay: {e}")))?;

        let _ = std::fs::remove_file(&audio_path);

        if !status.success() {
            return Err(SynthError::PlaybackFailed(format!(
                "afplay exited with {status}"
            )));
        }
    }

    Ok(())
}
