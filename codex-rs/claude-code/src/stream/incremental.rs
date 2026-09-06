use super::DecodeError;
use super::DecodedStream;
use super::PresentationDelta;
use super::state::Decoder;

/// Incrementally frames native Claude SSE bytes into the strict decoder state machine.
pub struct IncrementalDecoder<F> {
    decoder: Decoder<F>,
    frame: Vec<u8>,
    saw_event: bool,
}

impl<F> IncrementalDecoder<F>
where
    F: FnMut(PresentationDelta),
{
    pub fn new(sink: F) -> Self {
        Self {
            decoder: Decoder::new(sink),
            frame: Vec::new(),
            saw_event: false,
        }
    }

    pub fn feed(&mut self, bytes: &[u8]) -> Result<(), DecodeError> {
        self.frame.extend_from_slice(bytes);
        while let Some(end) = frame_end(&self.frame) {
            let frame = self.frame.drain(..end).collect::<Vec<_>>();
            self.consume_frame(&frame)?;
            while matches!(self.frame.first(), Some(b'\r' | b'\n')) {
                self.frame.remove(0);
            }
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<DecodedStream, DecodeError> {
        if !self.frame.is_empty() {
            let frame = std::mem::take(&mut self.frame);
            self.consume_frame(&frame)?;
        }
        if !self.saw_event {
            return Err(DecodeError::EmptyStream);
        }
        self.decoder.finish()
    }

    pub fn is_complete(&self) -> bool {
        self.decoder.is_stopped()
    }

    fn consume_frame(&mut self, bytes: &[u8]) -> Result<(), DecodeError> {
        let frame = std::str::from_utf8(bytes).map_err(|_| DecodeError::InvalidUtf8)?;
        let frame = frame.replace("\r\n", "\n").replace('\r', "\n");
        let mut event = None;
        let mut data = Vec::new();
        for line in frame.lines() {
            if line.is_empty() {
                continue;
            }
            let (field, value) = line
                .split_once(':')
                .ok_or_else(|| DecodeError::MalformedSse(format!("invalid line {line:?}")))?;
            let value = value.strip_prefix(' ').unwrap_or(value);
            match field {
                "event" if event.replace(value.to_string()).is_none() => {}
                "data" => data.push(value),
                "event" => {
                    return Err(DecodeError::MalformedSse(
                        "frame had duplicate event fields".to_string(),
                    ));
                }
                _ => {
                    return Err(DecodeError::MalformedSse(format!(
                        "unsupported SSE field {field}"
                    )));
                }
            }
        }
        let event = event
            .ok_or_else(|| DecodeError::MalformedSse("frame had data but no event".to_string()))?;
        if data.is_empty() {
            return Err(DecodeError::MalformedSse(format!(
                "{event} frame had no data"
            )));
        }
        self.saw_event = true;
        self.decoder.event(&event, &data.join("\n"))
    }
}

fn frame_end(bytes: &[u8]) -> Option<usize> {
    bytes
        .windows(2)
        .position(|window| window == b"\n\n" || window == b"\r\r")
        .map(|index| index + 2)
        .or_else(|| {
            bytes
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|index| index + 4)
        })
}
