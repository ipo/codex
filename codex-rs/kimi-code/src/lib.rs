//! Native Kimi Code request semantics layered on Chat Completions.

mod request;
mod schema;

pub use request::KimiDialect;
pub use request::KimiEncodeRequest;
pub use request::KimiError;
pub use request::KimiRequestSettings;
pub use request::KimiThinking;
pub use request::KimiThinkingEffort;
pub use request::encode_request;
pub use schema::SchemaError;
pub use schema::normalize_schema;

#[cfg(test)]
#[path = "request_tests.rs"]
mod tests;
