use std::{collections::HashMap, fs, path::Path};

use crate::error::VoiceError;

/// Loaded voice bank. Each voice maps to a flat f32 array of shape `[n_styles, 256]`
/// stored contiguously. `n_styles` differs per voice (typically ~510 entries).
#[derive(Debug)]
pub struct VoiceBank {
    voices: HashMap<String, Vec<f32>>,
}

impl VoiceBank {
    /// Load every `*.bin` file in `voices_dir`.
    /// File stem becomes the voice name (e.g. `af_heart.bin` → `"af_heart"`).
    pub fn load(voices_dir: &Path) -> Result<Self, VoiceError> {
        let mut voices = HashMap::new();

        let entries = fs::read_dir(voices_dir).map_err(|source| VoiceError::DirOpen {
            path: voices_dir.to_path_buf(),
            source,
        })?;

        for entry in entries {
            let entry = entry.map_err(VoiceError::DirEntry)?;
            let path = entry.path();

            if path.extension().and_then(|e| e.to_str()) != Some("bin") {
                continue;
            }

            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or_else(|| VoiceError::InvalidFilename(path.clone()))?
                .to_string();

            let raw = fs::read(&path).map_err(|source| VoiceError::FileRead {
                path: path.clone(),
                source,
            })?;

            if raw.len() % (256 * 4) != 0 {
                return Err(VoiceError::InvalidBinSize {
                    path: path.clone(),
                    len: raw.len(),
                });
            }

            // Reinterpret bytes as little-endian f32 (same as np.fromfile dtype=float32)
            let samples: Vec<f32> = raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();

            voices.insert(name, samples);
        }

        if voices.is_empty() {
            return Err(VoiceError::EmptyVoicesDir(voices_dir.to_path_buf()));
        }

        tracing::info!("loaded {} voice(s)", voices.len());
        Ok(Self { voices })
    }

    /// Returns a `[256]` style embedding for `voice`, chosen by `seq_len`.
    ///
    /// The voice tensor has shape `[n_styles, 256]`. We pick the row at
    /// `min(seq_len, n_styles - 1)` — the same heuristic used by the Python
    /// kokoro reference implementation.
    pub fn get_style(&self, voice: &str, seq_len: usize) -> Result<Vec<f32>, VoiceError> {
        let bank = self
            .voices
            .get(voice)
            .ok_or_else(|| VoiceError::UnknownVoice {
                name: voice.to_string(),
            })?;

        let n_styles = bank.len() / 256;
        if n_styles == 0 {
            return Err(VoiceError::ZeroStyles {
                name: voice.to_string(),
            });
        }

        let idx = seq_len.min(n_styles - 1);
        Ok(bank[idx * 256..(idx + 1) * 256].to_vec())
    }

    /// Return all loaded voice names, sorted.
    pub fn voice_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.voices.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    fn write_voice_bin(dir: &Path, name: &str, n_styles: usize) {
        let path = dir.join(format!("{name}.bin"));
        let floats: Vec<f32> = (0..n_styles * 256).map(|i| i as f32 * 0.001).collect();
        let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&bytes)
            .unwrap();
    }

    #[test]
    fn load_and_list_voices() {
        let dir = tempdir().unwrap();
        write_voice_bin(dir.path(), "af_heart", 10);
        write_voice_bin(dir.path(), "am_adam", 8);

        let bank = VoiceBank::load(dir.path()).unwrap();
        let names = bank.voice_names();
        assert_eq!(names, vec!["af_heart", "am_adam"]);
    }

    #[test]
    fn get_style_256_floats() {
        let dir = tempdir().unwrap();
        write_voice_bin(dir.path(), "af_heart", 20);

        let bank = VoiceBank::load(dir.path()).unwrap();
        let style = bank.get_style("af_heart", 5).unwrap();
        assert_eq!(style.len(), 256);
    }

    #[test]
    fn get_style_clamps_seq_len() {
        let dir = tempdir().unwrap();
        write_voice_bin(dir.path(), "af_heart", 4);

        let bank = VoiceBank::load(dir.path()).unwrap();
        let style = bank.get_style("af_heart", 9999).unwrap();
        assert_eq!(style.len(), 256);
        let expected_first = 3.0 * 256.0 * 0.001;
        assert!((style[0] - expected_first).abs() < 1e-4);
    }

    #[test]
    fn unknown_voice_variant() {
        let dir = tempdir().unwrap();
        write_voice_bin(dir.path(), "af_heart", 5);

        let bank = VoiceBank::load(dir.path()).unwrap();
        let err = bank.get_style("nonexistent", 0).unwrap_err();
        assert!(
            matches!(err, VoiceError::UnknownVoice { ref name } if name == "nonexistent"),
            "got {err:?}",
        );
    }

    #[test]
    fn empty_dir_error() {
        let dir = tempdir().unwrap();
        let err = VoiceBank::load(dir.path()).unwrap_err();
        assert!(matches!(err, VoiceError::EmptyVoicesDir(_)), "got {err:?}");
    }

    #[test]
    fn invalid_bin_variant() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bad.bin");
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&[0u8; 5])
            .unwrap();

        let err = VoiceBank::load(dir.path()).unwrap_err();
        assert!(
            matches!(err, VoiceError::InvalidBinSize { .. }),
            "got {err:?}",
        );
    }

    #[test]
    fn dir_open_has_path() {
        let err = VoiceError::DirOpen {
            path: "/missing/voices".into(),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "not found"),
        };
        assert!(err.to_string().contains("/missing/voices"));
    }

    #[test]
    fn unknown_voice_has_name() {
        let err = VoiceError::UnknownVoice {
            name: "af_mystery".into(),
        };
        assert!(err.to_string().contains("af_mystery"));
    }

    #[test]
    fn bin_size_error_details() {
        let err = VoiceError::InvalidBinSize {
            path: "/tmp/bad.bin".into(),
            len: 5,
        };
        let msg = err.to_string();
        assert!(msg.contains("/tmp/bad.bin"));
        assert!(msg.contains("5 bytes"));
    }
}
