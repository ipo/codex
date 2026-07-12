use std::env;
use std::path::Path;
use std::process::Command;

const BUILD_BRANCH_ENV: &str = "CODEX_BUILD_BRANCH";
const MAX_BRANCH_CHARS: usize = 128;
const DETACHED_SHA_LEN: usize = 12;

fn main() {
    println!("cargo:rerun-if-env-changed={BUILD_BRANCH_ENV}");

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

    println!("cargo:rustc-env={BUILD_BRANCH_ENV}={build_branch}");
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
