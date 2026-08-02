//! Native Kimi Code request semantics layered on Chat Completions.

mod request;
mod response;
mod schema;

pub use request::KimiDialect;
pub use request::KimiEncodeRequest;
pub use request::KimiError;
pub use request::KimiRequestSettings;
pub use request::KimiThinking;
pub use request::KimiThinkingEffort;
pub use request::encode_request;
pub use response::KimiResponseError;
pub use response::response_items;
pub use schema::SchemaError;
pub use schema::normalize_schema;

#[cfg(test)]
#[path = "request_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "response_tests.rs"]
mod response_tests;
