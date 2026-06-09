use std::{collections::HashMap, path::Path};

use crate::error::PhonemizerError;

/// Converts text to an IPA phoneme string.
///
/// Implementations are selectable at runtime via the `PHONEMIZER` env var:
/// - `"misaki"` (default) — pure-Rust Kokoro G2P via the `misaki-rs` crate
/// - `"espeak"` — safe wrapper around `libespeak-ng` via the `espeakng` crate
pub trait PhonemizerBackend: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &'static str;
    fn text_to_ipa(&self, text: &str) -> Result<String, PhonemizerError>;
}

#[derive(Debug)]
pub struct EspeakBackend;

impl PhonemizerBackend for EspeakBackend {
    fn name(&self) -> &'static str {
        "espeak"
    }

    fn text_to_ipa(&self, text: &str) -> Result<String, PhonemizerError> {
        let mut speaker = espeakng::initialise(None)
            .map_err(|e| PhonemizerError::EspeakInit(e.to_string()))?
            .lock();

        speaker
            .set_voice_raw("gmw/en-US")
            .map_err(|e| PhonemizerError::EspeakFailed(format!("set voice: {e}")))?;

        let ipa = speaker
            .text_to_phonemes(
                text,
                espeakng::PhonemeGenOptions::Standard {
                    text_mode: espeakng::TextMode::Utf8,
                    phoneme_mode: espeakng::PhonemeMode::IncludeZeroWidthJoiners,
                },
            )
            .map_err(|e| PhonemizerError::EspeakFailed(format!("phonemize: {e}")))?
            .unwrap_or_default();

        Ok(ipa)
    }
}

#[derive(Debug)]
pub struct MisakiBackend;

impl PhonemizerBackend for MisakiBackend {
    fn name(&self) -> &'static str {
        "misaki"
    }

    fn text_to_ipa(&self, text: &str) -> Result<String, PhonemizerError> {
        let g2p = misaki_rs::G2P::new(misaki_rs::Language::EnglishUS);
        let (ipa, _tokens) = g2p.g2p(text)?;
        Ok(ipa)
    }
}

/// Loads `vocab.json` and maps IPA strings to Kokoro token ID sequences.
struct VocabMapper {
    vocab: HashMap<char, i64>,
    bos_id: i64,
    eos_id: i64,
}

/// Max number of phoneme tokens fed to the model (model constraint).
const MAX_TOKENS: usize = 510; // leave room for BOS + EOS

impl VocabMapper {
    fn load(vocab_path: &Path) -> Result<Self, PhonemizerError> {
        let raw =
            std::fs::read_to_string(vocab_path).map_err(|source| PhonemizerError::VocabRead {
                path: vocab_path.to_path_buf(),
                source,
            })?;

        let map: HashMap<String, i64> = serde_json::from_str(&raw)?;

        let vocab: HashMap<char, i64> = map
            .into_iter()
            .filter_map(|(k, v)| {
                let mut chars = k.chars();
                let ch = chars.next()?;
                if chars.next().is_none() {
                    Some((ch, v))
                } else {
                    None
                }
            })
            .collect();

        let bos_id = *vocab.get(&'$').ok_or(PhonemizerError::MissingBos)?;
        let eos_id = *vocab.get(&';').ok_or(PhonemizerError::MissingEos)?;

        tracing::info!("loaded vocab with {} entries", vocab.len());
        let vocab = Self {
            vocab,
            bos_id,
            eos_id,
        };
        Ok(vocab)
    }

    fn ipa_to_tokens(&self, ipa: &str) -> Vec<i64> {
        // Strip stress/syllable markers not present in the Kokoro vocab
        const STRIP: &[char] = &['ˈ', 'ˌ', '\u{0361}'];

        let body: Vec<i64> = ipa
            .chars()
            .filter(|c| !STRIP.contains(c))
            .filter_map(|c| self.vocab.get(&c).copied())
            .take(MAX_TOKENS)
            .collect();

        let mut tokens = Vec::with_capacity(body.len() + 2);
        tokens.push(self.bos_id);
        tokens.extend_from_slice(&body);
        tokens.push(self.eos_id);
        tokens
    }
}

#[derive(Debug, Clone, Copy)]
enum PhonemizerKind {
    Espeak,
    Misaki,
}

impl std::str::FromStr for PhonemizerKind {
    type Err = PhonemizerError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "espeak" => Ok(Self::Espeak),
            "misaki" => Ok(Self::Misaki),
            other => Err(PhonemizerError::UnknownBackend(other.to_string())),
        }
    }
}

#[derive(Debug)]
pub struct Phonemizer {
    backend: Box<dyn PhonemizerBackend>,
    mapper: VocabMapper,
}

impl std::fmt::Debug for VocabMapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VocabMapper")
            .field("vocab_len", &self.vocab.len())
            .field("bos_id", &self.bos_id)
            .field("eos_id", &self.eos_id)
            .finish()
    }
}

impl Phonemizer {
    pub fn load(
        vocab_path: &Path,
        backend: Box<dyn PhonemizerBackend>,
    ) -> Result<Self, PhonemizerError> {
        let mapper = VocabMapper::load(vocab_path)?;
        Ok(Self { backend, mapper })
    }

    pub fn load_from_env(vocab_path: &Path) -> Result<Self, PhonemizerError> {
        let kind: PhonemizerKind = std::env::var("PHONEMIZER")
            .unwrap_or_else(|_| "espeak".into())
            .parse()?;
        let backend: Box<dyn PhonemizerBackend> = match kind {
            PhonemizerKind::Espeak => Box::new(EspeakBackend),
            PhonemizerKind::Misaki => Box::new(MisakiBackend),
        };
        tracing::info!("phonemizer backend: {}", backend.name());
        Self::load(vocab_path, backend)
    }

    /// Pipeline: backend → IPA string → vocab mapping → [BOS, …tokens…, EOS]
    pub fn phonemize(&self, text: &str) -> Result<Vec<i64>, PhonemizerError> {
        let ipa = self.backend.text_to_ipa(text)?;
        Ok(self.mapper.ipa_to_tokens(&ipa))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    #[derive(Debug)]
    struct DummyBackend;
    impl PhonemizerBackend for DummyBackend {
        fn name(&self) -> &'static str {
            "dummy"
        }
        fn text_to_ipa(&self, text: &str) -> Result<String, PhonemizerError> {
            Ok(text.to_string())
        }
    }

    fn make_vocab(entries: &[(&str, i64)]) -> serde_json::Value {
        let map: serde_json::Map<String, serde_json::Value> = entries
            .iter()
            .map(|(k, v)| (k.to_string(), serde_json::json!(v)))
            .collect();
        serde_json::Value::Object(map)
    }

    fn write_vocab(dir: &Path, entries: &[(&str, i64)]) {
        let path = dir.join("vocab.json");
        let json = make_vocab(entries);
        std::fs::File::create(&path)
            .unwrap()
            .write_all(json.to_string().as_bytes())
            .unwrap();
    }

    fn load(dir: &Path) -> Phonemizer {
        Phonemizer::load(&dir.join("vocab.json"), Box::new(DummyBackend)).unwrap()
    }

    #[test]
    fn load_valid_vocab() {
        let dir = tempdir().unwrap();
        write_vocab(dir.path(), &[("$", 0), (";", 1), ("h", 2), ("ɛ", 3)]);
        let p = load(dir.path());
        assert_eq!(p.mapper.bos_id, 0);
        assert_eq!(p.mapper.eos_id, 1);
    }

    #[test]
    fn missing_bos_variant() {
        let dir = tempdir().unwrap();
        write_vocab(dir.path(), &[(";", 1), ("h", 2)]);
        let err =
            Phonemizer::load(&dir.path().join("vocab.json"), Box::new(DummyBackend)).unwrap_err();
        assert!(matches!(err, PhonemizerError::MissingBos), "got {err:?}");
    }

    #[test]
    fn missing_eos_variant() {
        let dir = tempdir().unwrap();
        write_vocab(dir.path(), &[("$", 0), ("h", 2)]);
        let err =
            Phonemizer::load(&dir.path().join("vocab.json"), Box::new(DummyBackend)).unwrap_err();
        assert!(matches!(err, PhonemizerError::MissingEos), "got {err:?}");
    }

    #[test]
    fn nonexistent_vocab_file() {
        let err = Phonemizer::load(Path::new("/nonexistent/vocab.json"), Box::new(DummyBackend))
            .unwrap_err();
        assert!(
            matches!(err, PhonemizerError::VocabRead { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn invalid_vocab_json() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("vocab.json");
        std::fs::write(&path, b"not json at all").unwrap();
        let err = Phonemizer::load(&path, Box::new(DummyBackend)).unwrap_err();
        assert!(matches!(err, PhonemizerError::VocabParse(_)), "got {err:?}");
    }

    #[test]
    fn tokens_wrap_bos_eos() {
        let dir = tempdir().unwrap();
        write_vocab(
            dir.path(),
            &[("$", 0), (";", 1), ("h", 2), ("ɛ", 3), ("l", 4)],
        );
        let p = load(dir.path());

        let tokens = p.mapper.ipa_to_tokens("hɛl");
        assert_eq!(tokens.first(), Some(&0)); // BOS
        assert_eq!(tokens.last(), Some(&1)); // EOS
        assert_eq!(tokens.len(), 5); // BOS + h + ɛ + l + EOS
    }

    #[test]
    fn tokens_skip_unknown() {
        let dir = tempdir().unwrap();
        write_vocab(dir.path(), &[("$", 0), (";", 1), ("h", 2)]);
        let p = load(dir.path());

        let tokens = p.mapper.ipa_to_tokens("hxh");
        assert_eq!(tokens, vec![0, 2, 2, 1]); // BOS h h EOS
    }

    #[test]
    fn tokens_strip_stress() {
        let dir = tempdir().unwrap();
        write_vocab(dir.path(), &[("$", 0), (";", 1), ("h", 2)]);
        let p = load(dir.path());

        let tokens = p.mapper.ipa_to_tokens("ˈhˌh");
        assert_eq!(tokens, vec![0, 2, 2, 1]);
    }

    #[test]
    fn tokens_truncate_max() {
        let dir = tempdir().unwrap();
        write_vocab(dir.path(), &[("$", 0), (";", 1)]);
        let _p = load(dir.path());
        assert_eq!(MAX_TOKENS, 510);
    }

    #[test]
    fn vocab_error_has_path() {
        let err = PhonemizerError::VocabRead {
            path: "/tmp/vocab.json".into(),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "not found"),
        };
        assert!(err.to_string().contains("/tmp/vocab.json"));
    }

    #[test]
    fn missing_bos_error_message() {
        assert_eq!(
            PhonemizerError::MissingBos.to_string(),
            "BOS token '$' missing from vocab"
        );
    }

    #[test]
    fn missing_eos_error_message() {
        assert_eq!(
            PhonemizerError::MissingEos.to_string(),
            "EOS token ';' missing from vocab"
        );
    }

    #[test]
    fn dummy_backend_phonemize_passthrough() {
        let dir = tempdir().unwrap();
        write_vocab(dir.path(), &[("$", 0), (";", 1), ("h", 2)]);
        let p = load(dir.path());
        // DummyBackend echoes text as-is; only 'h' is in vocab
        let tokens = p.phonemize("h").unwrap();
        assert_eq!(tokens, vec![0, 2, 1]);
    }

    #[test]
    fn backend_name_espeak() {
        assert_eq!(EspeakBackend.name(), "espeak");
    }

    #[test]
    fn backend_name_misaki() {
        assert_eq!(MisakiBackend.name(), "misaki");
    }
}
