mod incremental;
#[path = "stream_state.rs"]
mod state;

pub use incremental::IncrementalDecoder;
pub use state::DecodeError;
pub use state::DecodedBlock;
pub use state::DecodedStream;
pub use state::PresentationDelta;
pub use state::decode_stream;
