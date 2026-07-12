use std::env;
use std::path::Path;
use std::process::Command;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

const BUILD_BRANCH_ENV: &str = "CODEX_BUILD_BRANCH";
const BUILD_TIMESTAMP_ENV: &str = "CODEX_BUILD_TIMESTAMP";
const SOURCE_DATE_EPOCH_ENV: &str = "SOURCE_DATE_EPOCH";
const MAX_BRANCH_CHARS: usize = 128;
const DETACHED_SHA_LEN: usize = 12;
const SECONDS_PER_DAY: i64 = 86_400;

fn main() {
    println!("cargo:rerun-if-env-changed={BUILD_BRANCH_ENV}");
    println!("cargo:rerun-if-env-changed={SOURCE_DATE_EPOCH_ENV}");

    let manifest_dir = env::var_os("CARGO_MANIFEST_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    emit_git_rerun_directives(&manifest_dir);

    let build_branch = env::var(BUILD_BRANCH_ENV)
        .ok()
        .and_then(|value| normalize_branch(&value))
        .or_else(|| {
            git_output(
                &manifest_dir,
                &["symbolic-ref", "--quiet", "--short", "HEAD"],
            )
        })
        .and_then(|value| normalize_branch(&value))
        .or_else(|| detached_head_label(&manifest_dir))
        .unwrap_or_else(|| "unknown".to_string());
    let build_timestamp = env::var(SOURCE_DATE_EPOCH_ENV)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .and_then(format_utc_timestamp)
        .or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .ok()
                .and_then(|duration| format_utc_timestamp(duration.as_secs()))
        })
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env={BUILD_BRANCH_ENV}={build_branch}");
    println!("cargo:rustc-env={BUILD_TIMESTAMP_ENV}={build_timestamp}");
}

fn format_utc_timestamp(epoch_seconds: u64) -> Option<String> {
    let epoch_seconds = i64::try_from(epoch_seconds).ok()?;
    let days_since_epoch = epoch_seconds.div_euclid(SECONDS_PER_DAY);
    let second_of_day = epoch_seconds.rem_euclid(SECONDS_PER_DAY);

    // Convert days since the Unix epoch to a proleptic Gregorian date using
    // Howard Hinnant's civil-from-days algorithm.
    let shifted_days = days_since_epoch + 719_468;
    let era = if shifted_days >= 0 {
        shifted_days
    } else {
        shifted_days - 146_096
    } / 146_097;
    let day_of_era = shifted_days - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    if !(0..=9_999).contains(&year) {
        return None;
    }

    let hour = second_of_day / 3_600;
    let minute = second_of_day % 3_600 / 60;
    let second = second_of_day % 60;
    Some(format!(
        "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z"
    ))
}

fn normalize_branch(value: &str) -> Option<String> {
    let value = value.trim();
    let value = value.strip_prefix("refs/heads/").unwrap_or(value);
    if value.is_empty() || value.chars().any(char::is_control) {
        return None;
    }

    Some(value.chars().take(MAX_BRANCH_CHARS).collect())
}

fn detached_head_label(manifest_dir: &Path) -> Option<String> {
    let commit = git_output(manifest_dir, &["rev-parse", "--verify", "HEAD"])?;
    let commit = commit.trim();
    if commit.len() < DETACHED_SHA_LEN || !commit.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    Some(format!(
        "detached-{}",
        commit[..DETACHED_SHA_LEN].to_ascii_lowercase()
    ))
}

fn emit_git_rerun_directives(manifest_dir: &Path) {
    if let Some(head_path) = git_path(manifest_dir, "HEAD") {
        println!("cargo:rerun-if-changed={}", head_path.display());
    }

    let Some(symbolic_ref) = git_output(manifest_dir, &["symbolic-ref", "--quiet", "HEAD"]) else {
        return;
    };
    if let Some(ref_path) = git_path(manifest_dir, symbolic_ref.trim()) {
        println!("cargo:rerun-if-changed={}", ref_path.display());
    }
}

fn git_path(manifest_dir: &Path, path: &str) -> Option<std::path::PathBuf> {
    git_output(
        manifest_dir,
        &["rev-parse", "--path-format=absolute", "--git-path", path],
    )
    .map(std::path::PathBuf::from)
}

fn git_output(manifest_dir: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(manifest_dir)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    String::from_utf8(output.stdout)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}
