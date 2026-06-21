//! Phase 4.4 — argv parsing for `sparrow-engine-server`.
//!
//! `sparrow-engine-server` is configured entirely through `SPARROW_ENGINE_*`
//! environment variables (see [`crate::config::Config::from_env`]). The only
//! CLI surface is:
//!
//! - `sparrow-engine-server` (no args) — boot the HTTP server (Docker
//!   `ENTRYPOINT`).
//! - `sparrow-engine-server healthcheck` — issue a local health check (Docker
//!   `HEALTHCHECK CMD`).
//! - `sparrow-engine-server --help` / `-h` — print help and exit 0.
//! - `sparrow-engine-server --version` / `-V` — print version and exit 0.
//!
//! Before Phase 4.4 the binary recognized only `healthcheck` and silently
//! fell through every other argument (including `--help` / `--version`) to
//! the full server-boot path. On a shared multi-tenant host this leaked
//! lingering 129-thread processes pinned to a deleted binary inode
//! (MT-4.1-26). Routing argv through `clap` here resolves that hazard and
//! keeps the existing `healthcheck` subcommand intact.

use clap::{Parser, Subcommand};

/// HTTP server for Sparrow Engine inference.
#[derive(Debug, Parser)]
#[command(
    name = "sparrow-engine-server",
    version,
    about = "HTTP server for Sparrow Engine inference",
    long_about = LONG_ABOUT,
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,
}

/// Subcommands recognized by `sparrow-engine-server`.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run a local health check against this server (used by Docker HEALTHCHECK).
    Healthcheck,
}

const LONG_ABOUT: &str = "\
HTTP server for Sparrow Engine inference.

USAGE
  sparrow-engine-server                Boot the HTTP server.
  sparrow-engine-server healthcheck    Issue a local health check against this server
                                       (used by Docker HEALTHCHECK CMD).

Configuration is read entirely from environment variables. The supported
variables are:

  SPARROW_ENGINE_BIND_ADDR                 Socket address to bind                [default: 0.0.0.0:8080]
  SPARROW_ENGINE_MODEL_DIR                 Directory containing model manifests  [default: /models]
  SPARROW_ENGINE_LOG_FORMAT                'json' or 'pretty'                    [default: json]
  SPARROW_ENGINE_LOG_LEVEL                 tracing filter directive              [default: info]
  SPARROW_ENGINE_DEVICE                    'auto', 'cpu', or 'cuda:N'            [default: auto]
  SPARROW_ENGINE_PRELOAD                   Comma-separated model ids, or 'all', to eager-load at boot [optional]
  SPARROW_ENGINE_MAX_BODY_SIZE             Max request body size, e.g. 100mb     [default: 100mb]
  SPARROW_ENGINE_MAX_CONCURRENT_INFERENCE  Concurrency limit                     [default: 32]
  SPARROW_ENGINE_MAX_BATCH_SIZE            Max batch size                        [default: 64]
  SPARROW_ENGINE_REQUEST_TIMEOUT           Per-request timeout, seconds          [default: 120]
  SPARROW_ENGINE_DRAIN_TIMEOUT             Graceful-shutdown drain, seconds      [default: 10]
  SPARROW_ENGINE_INTER_THREADS             ORT inter-op threads (u32)            [optional]
  SPARROW_ENGINE_INTRA_THREADS             ORT intra-op threads (u32)            [optional]

See docs/master_plan.md and docs/design/phase4.2-cold-start/ for the full
operator reference.";

#[cfg(test)]
mod tests {
    use super::*;
    use clap::error::ErrorKind;

    #[test]
    fn no_args_parses_as_serve() {
        let cli = Cli::try_parse_from(["sparrow-engine-server"]).expect("clap parse");
        assert!(cli.command.is_none());
    }

    #[test]
    fn healthcheck_subcommand_parses() {
        let cli =
            Cli::try_parse_from(["sparrow-engine-server", "healthcheck"]).expect("clap parse");
        assert!(matches!(cli.command, Some(Command::Healthcheck)));
    }

    #[test]
    fn long_help_flag_emits_display_help() {
        let err = Cli::try_parse_from(["sparrow-engine-server", "--help"])
            .expect_err("expected DisplayHelp");
        assert_eq!(err.kind(), ErrorKind::DisplayHelp);
    }

    #[test]
    fn short_help_flag_emits_display_help() {
        let err =
            Cli::try_parse_from(["sparrow-engine-server", "-h"]).expect_err("expected DisplayHelp");
        assert_eq!(err.kind(), ErrorKind::DisplayHelp);
    }

    #[test]
    fn long_version_flag_emits_display_version() {
        let err = Cli::try_parse_from(["sparrow-engine-server", "--version"])
            .expect_err("expected DisplayVersion");
        assert_eq!(err.kind(), ErrorKind::DisplayVersion);
    }

    #[test]
    fn short_version_flag_emits_display_version() {
        let err = Cli::try_parse_from(["sparrow-engine-server", "-V"])
            .expect_err("expected DisplayVersion");
        assert_eq!(err.kind(), ErrorKind::DisplayVersion);
    }

    #[test]
    fn help_rendering_lists_known_env_vars() {
        // Render the long-help text and confirm every documented env var is
        // present. Guards against accidental drift between Config::from_env
        // and the help docstring.
        let err =
            Cli::try_parse_from(["sparrow-engine-server", "--help"]).expect_err("DisplayHelp");
        let rendered = err.render().to_string();
        for var in [
            "SPARROW_ENGINE_BIND_ADDR",
            "SPARROW_ENGINE_MODEL_DIR",
            "SPARROW_ENGINE_LOG_FORMAT",
            "SPARROW_ENGINE_LOG_LEVEL",
            "SPARROW_ENGINE_DEVICE",
            "SPARROW_ENGINE_PRELOAD",
            "SPARROW_ENGINE_MAX_BODY_SIZE",
            "SPARROW_ENGINE_MAX_CONCURRENT_INFERENCE",
            "SPARROW_ENGINE_MAX_BATCH_SIZE",
            "SPARROW_ENGINE_REQUEST_TIMEOUT",
            "SPARROW_ENGINE_DRAIN_TIMEOUT",
            "SPARROW_ENGINE_INTER_THREADS",
            "SPARROW_ENGINE_INTRA_THREADS",
        ] {
            assert!(
                rendered.contains(var),
                "help text missing env var {var}; full text was:\n{rendered}"
            );
        }
    }

    #[test]
    fn unknown_arg_is_an_error() {
        let err = Cli::try_parse_from(["sparrow-engine-server", "definitely-not-a-real-arg"])
            .expect_err("error");
        // clap classifies positional args that don't match a known subcommand
        // as InvalidSubcommand; unknown long/short flags become UnknownArgument.
        assert!(
            matches!(
                err.kind(),
                ErrorKind::InvalidSubcommand | ErrorKind::UnknownArgument
            ),
            "unexpected error kind: {:?}",
            err.kind()
        );
    }

    #[test]
    fn unknown_flag_is_an_error() {
        let err =
            Cli::try_parse_from(["sparrow-engine-server", "--bogus-flag"]).expect_err("error");
        assert_eq!(err.kind(), ErrorKind::UnknownArgument);
    }
}
