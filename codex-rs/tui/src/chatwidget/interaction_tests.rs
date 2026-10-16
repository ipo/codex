use super::*;

#[cfg(target_os = "linux")]
struct ReapOnDrop(std::process::Child);

#[cfg(target_os = "linux")]
impl Drop for ReapOnDrop {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[tokio::test]
async fn repeated_paste_keeps_one_pending_reader() {
    let (mut chat, _tx, _rx, _op_rx) =
        crate::chatwidget::tests::make_chatwidget_manual_with_sender().await;
    let request_id = uuid::Uuid::new_v4();
    chat.clipboard_image_task = Some((request_id, tokio::spawn(std::future::pending())));
    chat.handle_key_event(KeyEvent::new(KeyCode::Char('v'), KeyModifiers::CONTROL));
    chat.handle_paste("x".to_string());
    assert_eq!(
        chat.clipboard_image_task.as_ref().map(|(id, _)| *id),
        Some(request_id)
    );
    assert!(chat.composer_text_with_pending().contains('x'));
}

#[tokio::test]
async fn result_for_different_conversation_is_discarded_and_file_removed() {
    let (mut chat, _tx, _rx, _op_rx) =
        crate::chatwidget::tests::make_chatwidget_manual_with_sender().await;
    let request_id = uuid::Uuid::new_v4();
    chat.clipboard_image_task = Some((request_id, tokio::spawn(std::future::pending())));
    let temp = tempfile::NamedTempFile::new().unwrap();
    let (_, path) = temp.keep().unwrap();
    chat.finish_clipboard_image_read(
        request_id,
        Some(ThreadId::new()),
        Ok((
            path.clone(),
            crate::clipboard_paste::PastedImageInfo {
                width: 1,
                height: 1,
                encoded_format: crate::clipboard_paste::EncodedImageFormat::Png,
            },
        )),
    );
    assert!(!path.exists());
    assert!(chat.clipboard_image_task.is_none());
}

#[tokio::test]
async fn aborted_waiter_cleans_late_native_image() {
    let directory = tempfile::tempdir().unwrap();
    let output_dir = directory.path().to_path_buf();
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let (path_tx, path_rx) = tokio::sync::oneshot::channel();
    let waiter = tokio::spawn(async move {
        let _ = spawn_native_clipboard_read(move || {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            let temp = tempfile::NamedTempFile::new_in(output_dir).unwrap();
            let (_, path) = temp.keep().unwrap();
            path_tx.send(path.clone()).unwrap();
            Ok((
                path,
                crate::clipboard_paste::PastedImageInfo {
                    width: 1,
                    height: 1,
                    encoded_format: crate::clipboard_paste::EncodedImageFormat::Png,
                },
            ))
        })
        .await;
    });
    started_rx.await.unwrap();
    waiter.abort();
    assert!(waiter.await.unwrap_err().is_cancelled());
    release_tx.send(()).unwrap();
    let path = path_rx.await.unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while path.exists() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("late native clipboard file was not removed after cancellation");
}

#[cfg(target_os = "linux")]
#[tokio::test]
#[ignore = "requires Xvfb and xclip; run explicitly in an isolated X11 session"]
async fn bmp_only_x11_clipboard_attaches_once() {
    use std::io::BufRead;
    use std::process::Stdio;

    if std::env::var_os("CODEX_BMP_X11_TEST_CHILD").is_some() {
        let (mut chat, _tx, mut rx, _op_rx) =
            crate::chatwidget::tests::make_chatwidget_manual_with_sender().await;
        chat.handle_key_event(KeyEvent::new(KeyCode::Char('v'), KeyModifiers::CONTROL));
        assert!(
            chat.clipboard_image_task.is_some(),
            "Ctrl+V must schedule clipboard work"
        );
        let event = tokio::time::timeout(std::time::Duration::from_secs(40), rx.recv())
            .await
            .unwrap_or_else(|_| {
                panic!(
                    "clipboard task did not return; finished={}",
                    chat.clipboard_image_task.as_ref().unwrap().1.is_finished()
                )
            })
            .unwrap();
        let AppEvent::ClipboardImageRead {
            request_id,
            thread_id,
            result,
        } = event
        else {
            panic!("expected clipboard image result");
        };
        chat.finish_clipboard_image_read(request_id, thread_id, result);
        let attachments = chat.bottom_pane.composer_local_image_paths();
        assert_eq!(attachments.len(), 1);
        let decoded = image::ImageReader::open(&attachments[0])
            .unwrap()
            .decode()
            .unwrap();
        assert_eq!((decoded.width(), decoded.height()), (1698, 2448));
        std::fs::remove_file(&attachments[0]).unwrap();
        return;
    }

    let directory = tempfile::tempdir().unwrap();
    let bmp = directory.path().join("source.bmp");
    image::DynamicImage::new_rgba8(1698, 2448)
        .save_with_format(&bmp, image::ImageFormat::Bmp)
        .unwrap();
    assert!(std::fs::metadata(&bmp).unwrap().len() > 16_000_000);

    let mut xvfb = ReapOnDrop(
        std::process::Command::new("Xvfb")
            .args(["-displayfd", "1", "-screen", "0", "1024x768x24"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("Xvfb is needed for the isolated X11 clipboard test"),
    );
    let mut display_number = String::new();
    std::io::BufReader::new(xvfb.0.stdout.take().unwrap())
        .read_line(&mut display_number)
        .unwrap();
    let display = format!(":{}", display_number.trim());
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let owner = std::thread::spawn(move || serve_bmp_clipboard(&display, &bmp, ready_tx));
    ready_rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("BMP-only X11 clipboard owner failed to start");

    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "chatwidget::interaction::tests::bmp_only_x11_clipboard_attaches_once",
            "--ignored",
            "--nocapture",
        ])
        .env("DISPLAY", format!(":{}", display_number.trim()))
        .env("CODEX_BMP_X11_TEST_CHILD", "1")
        .output()
        .unwrap();
    assert!(
        child.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&child.stdout),
        String::from_utf8_lossy(&child.stderr)
    );
    drop(xvfb);
    owner.join().unwrap();
}

#[cfg(target_os = "linux")]
fn serve_bmp_clipboard(
    display: &str,
    bmp_path: &std::path::Path,
    ready: std::sync::mpsc::Sender<()>,
) {
    use x11rb::connection::Connection;
    use x11rb::protocol::Event;
    use x11rb::protocol::xproto::AtomEnum;
    use x11rb::protocol::xproto::ConnectionExt;
    use x11rb::protocol::xproto::CreateWindowAux;
    use x11rb::protocol::xproto::EventMask;
    use x11rb::protocol::xproto::PropMode;
    use x11rb::protocol::xproto::Property;
    use x11rb::protocol::xproto::SelectionNotifyEvent;
    use x11rb::protocol::xproto::WindowClass;
    use x11rb::wrapper::ConnectionExt as _;

    let (conn, screen) = x11rb::connect(Some(display)).unwrap();
    let window = conn.generate_id().unwrap();
    conn.create_window(
        0,
        window,
        conn.setup().roots[screen].root,
        0,
        0,
        1,
        1,
        0,
        WindowClass::INPUT_ONLY,
        0,
        &CreateWindowAux::new(),
    )
    .unwrap();
    let atom = |name: &[u8]| conn.intern_atom(false, name).unwrap().reply().unwrap().atom;
    let clipboard = atom(b"CLIPBOARD");
    let targets = atom(b"TARGETS");
    let bmp = atom(b"image/bmp");
    let incr = atom(b"INCR");
    let bytes = std::fs::read(bmp_path).unwrap();
    conn.set_selection_owner(window, clipboard, x11rb::CURRENT_TIME)
        .unwrap();
    conn.flush().unwrap();
    ready.send(()).unwrap();

    // An INCR transfer advances only when the requestor deletes the previous chunk.
    let mut transfer: Option<(u32, u32, usize)> = None;
    // An X connection error means the parent test shut down Xvfb.
    while let Ok(event) = conn.wait_for_event() {
        match event {
            Event::SelectionRequest(request) => {
                let property = if request.property == u32::from(AtomEnum::NONE) {
                    request.target
                } else {
                    request.property
                };
                let mut notified_property = property;
                if request.target == targets {
                    conn.change_property32(
                        PropMode::REPLACE,
                        request.requestor,
                        property,
                        AtomEnum::ATOM,
                        &[targets, bmp],
                    )
                    .unwrap();
                } else if request.target == bmp {
                    conn.change_window_attributes(
                        request.requestor,
                        &x11rb::protocol::xproto::ChangeWindowAttributesAux::new()
                            .event_mask(EventMask::PROPERTY_CHANGE),
                    )
                    .unwrap();
                    conn.change_property32(
                        PropMode::REPLACE,
                        request.requestor,
                        property,
                        incr,
                        &[bytes.len().try_into().unwrap()],
                    )
                    .unwrap();
                    transfer = Some((request.requestor, property, 0));
                } else {
                    notified_property = AtomEnum::NONE.into();
                }
                conn.send_event(
                    false,
                    request.requestor,
                    EventMask::NO_EVENT,
                    SelectionNotifyEvent {
                        response_type: x11rb::protocol::xproto::SELECTION_NOTIFY_EVENT,
                        sequence: 0,
                        time: request.time,
                        requestor: request.requestor,
                        selection: request.selection,
                        target: request.target,
                        property: notified_property,
                    },
                )
                .unwrap();
                conn.flush().unwrap();
            }
            Event::PropertyNotify(event)
                if event.state == Property::DELETE
                    && transfer.as_ref().is_some_and(|(window, property, _)| {
                        event.window == *window && event.atom == *property
                    }) =>
            {
                let (requestor, property, offset) = transfer.unwrap();
                let end = (offset + 64 * 1024).min(bytes.len());
                conn.change_property8(
                    PropMode::REPLACE,
                    requestor,
                    property,
                    bmp,
                    &bytes[offset..end],
                )
                .unwrap();
                transfer = (offset < bytes.len()).then_some((requestor, property, end));
                conn.flush().unwrap();
            }
            _ => {}
        }
    }
}
