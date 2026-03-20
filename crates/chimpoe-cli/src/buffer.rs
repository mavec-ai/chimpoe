use anyhow::Result;
use chimpoe::types::Dialogue;
use std::fs;
use std::path::PathBuf;

pub fn buffer_path() -> PathBuf {
    crate::config::chimpoe_dir().join("buffer.json")
}

pub fn load() -> Result<Vec<Dialogue>> {
    let path = buffer_path();
    if !path.exists() {
        return Ok(Vec::new());
    }

    let content = fs::read_to_string(&path)?;
    let dialogues: Vec<Dialogue> = serde_json::from_str(&content)?;
    Ok(dialogues)
}

pub fn save(dialogues: &[Dialogue]) -> Result<()> {
    let path = buffer_path();
    let content = serde_json::to_string_pretty(dialogues)?;
    fs::write(path, content)?;
    Ok(())
}

pub fn clear() -> Result<()> {
    let path = buffer_path();
    if path.exists() {
        fs::remove_file(path)?;
    }
    Ok(())
}
