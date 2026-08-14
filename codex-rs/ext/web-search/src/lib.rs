mod extension;
mod history;
mod output;
mod schema;
mod tool;

pub use extension::install;
#[cfg(feature = "test-support")]
pub use extension::install_with_openai_base_url;
