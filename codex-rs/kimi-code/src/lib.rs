//! Native Kimi Code request semantics layered on Chat Completions.

mod request;
mod response;
mod schema;
mod transport;

pub use request::KimiDialect;
pub use request::KimiEncodeRequest;
pub use request::KimiError;
pub use request::KimiInputEstimate;
pub use request::KimiRequestSettings;
pub use request::KimiThinking;
pub use request::KimiThinkingEffort;
pub use request::encode_request;
pub use request::estimated_input_tokens;
pub use response::KimiResponseError;
pub use response::response_items;
pub use schema::SchemaError;
pub use schema::normalize_schema;
pub use transport::KimiHttpAdapter;
pub use transport::KimiResponseStream;
pub use transport::KimiStreamError;

#[cfg(test)]
#[path = "request_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "response_tests.rs"]
mod response_tests;

#[cfg(test)]
#[path = "transport_tests.rs"]
mod transport_tests;
