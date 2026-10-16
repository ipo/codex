//! Bounded X11 BMP clipboard fallback; native formats always take precedence.

use super::EncodedImageFormat;
use super::PasteImageError;
use super::PastedImageInfo;
use std::io::Cursor;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;
use tempfile::Builder;
use tokio::io::AsyncReadExt;
use tokio::process::Command;

const MAX_BMP_BYTES: usize = 64 * 1024 * 1024;
const MAX_DECODED_BYTES: u64 = 256 * 1024 * 1024;
const DEADLINE: Duration = Duration::from_secs(60);

pub(crate) fn should_try(error: &PasteImageError) -> bool {
    matches!(error, PasteImageError::NoImage(_))
        && std::env::var_os("DISPLAY").is_some()
        && !super::is_probably_wsl()
}

pub(crate) async fn read_xclip_bmp() -> Result<(PathBuf, PastedImageInfo), PasteImageError> {
    read_xclip_bmp_with("xclip", DEADLINE).await
}

async fn read_xclip_bmp_with(
    executable: &str,
    deadline: Duration,
) -> Result<(PathBuf, PastedImageInfo), PasteImageError> {
    let mut child = Command::new(executable)
        .args(["-selection", "clipboard", "-o", "-t", "image/bmp"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        // xclip diagnostics are not needed for paste errors; never buffer unbounded stderr.
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                PasteImageError::ClipboardUnavailable(
                    "BMP clipboard fallback requires xclip; install xclip to paste BMP images on X11"
                        .into(),
                )
            } else {
                PasteImageError::ClipboardUnavailable(format!("could not start xclip: {e}"))
            }
        })?;

    let read = async {
        let mut stdout = child.stdout.take().expect("piped stdout");
        let mut bmp = Vec::new();
        // The extra byte distinguishes an exactly-at-limit image from an oversized one.
        let mut bounded = (&mut stdout).take((MAX_BMP_BYTES + 1) as u64);
        bounded
            .read_to_end(&mut bmp)
            .await
            .map_err(|e| PasteImageError::IoError(e.to_string()))?;
        if bmp.len() > MAX_BMP_BYTES {
            return Err(PasteImageError::NoImage(
                "BMP clipboard image exceeds the 64 MiB limit".into(),
            ));
        }
        let status = child
            .wait()
            .await
            .map_err(|e| PasteImageError::IoError(e.to_string()))?;
        if !status.success() || bmp.is_empty() {
            return Err(PasteImageError::NoImage(
                "xclip did not provide a BMP image".into(),
            ));
        }
        Ok(bmp)
    };
    let bmp = match tokio::time::timeout(deadline, read).await {
        Ok(Ok(bmp)) => bmp,
        Ok(Err(error)) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            return Err(error);
        }
        Err(_) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            return Err(PasteImageError::ClipboardUnavailable(
                "BMP clipboard read timed out after 60 seconds; check the X11 clipboard owner and try again"
                    .into(),
            ));
        }
    };

    let (temp, info) = tokio::task::spawn_blocking(move || decode_to_temp_png(&bmp))
        .await
        .map_err(|e| PasteImageError::IoError(e.to_string()))??;
    let (_, path) = temp
        .keep()
        .map_err(|e| PasteImageError::IoError(e.error.to_string()))?;
    Ok((path, info))
}

fn decode_to_temp_png(
    bmp: &[u8],
) -> Result<(tempfile::NamedTempFile, PastedImageInfo), PasteImageError> {
    let mut reader = image::ImageReader::with_format(Cursor::new(bmp), image::ImageFormat::Bmp);
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(8192);
    limits.max_image_height = Some(8192);
    limits.max_alloc = Some(MAX_DECODED_BYTES);
    reader.limits(limits);
    let image = reader
        .decode()
        .map_err(|_| PasteImageError::NoImage("invalid or oversized BMP clipboard image".into()))?;
    let temp = Builder::new()
        .prefix("codex-clipboard-")
        .suffix(".png")
        .tempfile()
        .map_err(|e| PasteImageError::IoError(e.to_string()))?;
    image
        .save_with_format(temp.path(), image::ImageFormat::Png)
        .map_err(|e| PasteImageError::EncodeFailed(e.to_string()))?;
    Ok((
        temp,
        PastedImageInfo {
            width: image.width(),
            height: image.height(),
            encoded_format: EncodedImageFormat::Png,
        },
    ))
}

#[cfg(test)]
#[path = "x11_bmp_tests.rs"]
mod tests;
