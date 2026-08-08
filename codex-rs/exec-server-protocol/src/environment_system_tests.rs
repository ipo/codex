use pretty_assertions::assert_eq;

use super::EnvironmentOperatingSystem;
use super::EnvironmentSystemInfo;
use crate::EnvironmentCapabilities;
use crate::EnvironmentInfo;
use crate::ShellInfo;
use codex_utils_path_uri::PathUri;

#[test]
fn environment_info_round_trips_target_system_facts() {
    let info = EnvironmentInfo {
        shell: ShellInfo {
            name: "powershell".to_string(),
            path: "powershell.exe".to_string(),
        },
        cwd: Some(PathUri::parse("file:///C:/workspace").expect("Windows cwd URI")),
        capabilities: EnvironmentCapabilities::default(),
        system: Some(EnvironmentSystemInfo {
            operating_system: EnvironmentOperatingSystem::Windows,
            architecture: "x86_64".to_string(),
            os_version: "Windows 11".to_string(),
        }),
    };

    let encoded = serde_json::to_value(&info).expect("serialize environment info");
    let decoded: EnvironmentInfo =
        serde_json::from_value(encoded).expect("deserialize environment info");

    assert_eq!(decoded, info);
}
