use std::sync::OnceLock;

use codex_utils_path_uri::PathConvention;
use serde::Deserialize;
use serde::Serialize;

/// Authoritative operating-system identity reported by the execution target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum EnvironmentOperatingSystem {
    Linux,
    Macos,
    Windows,
}

impl EnvironmentOperatingSystem {
    /// Returns the stable platform name used in model-visible environment descriptions.
    pub fn platform(self) -> &'static str {
        match self {
            Self::Linux => "linux",
            Self::Macos => "macos",
            Self::Windows => "windows",
        }
    }

    /// Returns the native path convention for this operating system.
    pub fn path_convention(self) -> PathConvention {
        match self {
            Self::Linux | Self::Macos => PathConvention::Posix,
            Self::Windows => PathConvention::Windows,
        }
    }
}

/// System facts detected by the exec-server on the execution target.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnvironmentSystemInfo {
    pub operating_system: EnvironmentOperatingSystem,
    /// Rust target architecture name, for example `x86_64` or `aarch64`.
    pub architecture: String,
    /// Human-readable operating-system version detected on the execution target.
    pub os_version: String,
}

impl EnvironmentSystemInfo {
    pub(crate) fn local() -> Option<Self> {
        let operating_system = match std::env::consts::OS {
            "linux" => EnvironmentOperatingSystem::Linux,
            "macos" => EnvironmentOperatingSystem::Macos,
            "windows" => EnvironmentOperatingSystem::Windows,
            _ => return None,
        };
        Some(Self {
            operating_system,
            architecture: std::env::consts::ARCH.to_string(),
            os_version: local_os_version(),
        })
    }
}

fn local_os_version() -> String {
    static OS_VERSION: OnceLock<String> = OnceLock::new();
    OS_VERSION
        .get_or_init(|| {
            #[cfg(target_os = "linux")]
            if let Ok(version) = std::fs::read_to_string("/proc/sys/kernel/osrelease") {
                return format!("Linux {}", version.trim());
            }
            #[cfg(target_os = "macos")]
            if let Ok(output) = std::process::Command::new("sw_vers")
                .arg("-productVersion")
                .output()
                && output.status.success()
            {
                return format!("macOS {}", String::from_utf8_lossy(&output.stdout).trim());
            }
            #[cfg(windows)]
            if let Ok(output) = std::process::Command::new("cmd")
                .args(["/C", "ver"])
                .output()
                && output.status.success()
            {
                return String::from_utf8_lossy(&output.stdout).trim().to_string();
            }
            std::env::consts::OS.to_string()
        })
        .clone()
}

#[cfg(test)]
#[path = "environment_system_tests.rs"]
mod tests;
