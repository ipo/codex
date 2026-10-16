use super::*;
use pretty_assertions::assert_eq;
use std::os::unix::fs::PermissionsExt;

fn fake_xclip(directory: &tempfile::TempDir, script: &str) -> String {
    let executable = directory.path().join("fake-xclip");
    std::fs::write(&executable, format!("#!/bin/sh\n{script}\n")).unwrap();
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
    executable.to_str().unwrap().to_string()
}

#[tokio::test]
async fn delayed_bmp_is_encoded_as_png() {
    let directory = tempfile::tempdir().unwrap();
    let bmp = directory.path().join("source.bmp");
    image::DynamicImage::new_rgba8(32, 24)
        .save_with_format(&bmp, image::ImageFormat::Bmp)
        .unwrap();
    let command = fake_xclip(
        &directory,
        &format!("sleep 0.05\nexec /bin/cat {}", bmp.display()),
    );
    let (path, info) = read_xclip_bmp_with(&command, Duration::from_secs(2))
        .await
        .unwrap();
    assert_eq!(
        (info.width, info.height, info.encoded_format),
        (32, 24, EncodedImageFormat::Png)
    );
    let decoded = image::ImageReader::open(&path).unwrap().decode().unwrap();
    assert_eq!((decoded.width(), decoded.height()), (32, 24));
    std::fs::remove_file(path).unwrap();
}

#[tokio::test]
async fn large_bmp_clipboard_image_is_usable() {
    let directory = tempfile::tempdir().unwrap();
    let bmp = directory.path().join("source.bmp");
    image::DynamicImage::new_rgba8(1698, 2448)
        .save_with_format(&bmp, image::ImageFormat::Bmp)
        .unwrap();
    assert!(std::fs::metadata(&bmp).unwrap().len() > 16_000_000);
    let command = fake_xclip(&directory, &format!("exec /bin/cat {}", bmp.display()));
    let (png, info) = read_xclip_bmp_with(&command, Duration::from_secs(30))
        .await
        .unwrap();
    let decoded = image::ImageReader::open(&png).unwrap().decode().unwrap();
    assert_eq!(
        (decoded.width(), decoded.height()),
        (info.width, info.height)
    );
    std::fs::remove_file(png).unwrap();
}

#[tokio::test]
async fn deadline_kills_and_reaps_reader() {
    let directory = tempfile::tempdir().unwrap();
    let marker = directory.path().join("finished");
    let command = fake_xclip(&directory, &format!("sleep 1\ntouch {}", marker.display()));
    let result = read_xclip_bmp_with(&command, Duration::from_millis(30)).await;
    insta::assert_snapshot!(format!("Failed to paste image: {}", result.as_ref().unwrap_err()), @"Failed to paste image: clipboard unavailable: BMP clipboard read timed out after 60 seconds; check the X11 clipboard owner and try again");
    assert!(
        matches!(result, Err(PasteImageError::ClipboardUnavailable(message)) if message.contains("timed out"))
    );
    // The shell process has been reaped; no task-owned image file was created.
    assert!(!marker.exists());
}

#[tokio::test]
async fn cancelled_read_terminates_process() {
    let directory = tempfile::tempdir().unwrap();
    let pid_file = directory.path().join("pid");
    let command = fake_xclip(
        &directory,
        &format!("echo $$ > {}\nexec sleep 5", pid_file.display()),
    );
    let task =
        tokio::spawn(async move { read_xclip_bmp_with(&command, Duration::from_secs(10)).await });
    tokio::time::timeout(Duration::from_secs(2), async {
        while !pid_file.exists() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let pid = std::fs::read_to_string(&pid_file).unwrap();
    task.abort();
    let _ = task.await;
    tokio::time::timeout(Duration::from_secs(2), async {
        while std::path::Path::new(&format!("/proc/{}/cmdline", pid.trim())).exists() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn missing_executable_is_actionable() {
    let result = read_xclip_bmp_with("/nonexistent/codex-xclip", Duration::from_millis(20)).await;
    insta::assert_snapshot!(format!("Failed to paste image: {}", result.as_ref().unwrap_err()), @"Failed to paste image: clipboard unavailable: BMP clipboard fallback requires xclip; install xclip to paste BMP images on X11");
    assert!(
        matches!(result, Err(PasteImageError::ClipboardUnavailable(message)) if message.contains("install xclip"))
    );
}

#[tokio::test]
async fn malformed_and_oversized_images_are_rejected() {
    let directory = tempfile::tempdir().unwrap();
    let invalid = directory.path().join("invalid.bmp");
    std::fs::write(&invalid, b"not a BMP").unwrap();
    let command = fake_xclip(&directory, &format!("exec /bin/cat {}", invalid.display()));
    assert!(matches!(
        read_xclip_bmp_with(&command, Duration::from_secs(2)).await,
        Err(PasteImageError::NoImage(_))
    ));

    let oversized = directory.path().join("oversized.bmp");
    std::fs::write(&oversized, vec![0; MAX_BMP_BYTES + 2]).unwrap();
    let command = fake_xclip(
        &directory,
        &format!("exec /bin/cat {}", oversized.display()),
    );
    assert!(
        matches!(read_xclip_bmp_with(&command, Duration::from_secs(2)).await, Err(PasteImageError::NoImage(message)) if message.contains("64 MiB"))
    );
}
