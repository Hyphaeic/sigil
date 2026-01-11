//! # lib-ibis
//!
//! IBIS, AMI, and Touchstone file parsers for SI-Kernel.
//!
//! This crate provides parsers for:
//! - `.ibs` files (IBIS I/O buffer models)
//! - `.ami` files (AMI parameter files)
//! - `.sNp` files (Touchstone S-parameters)
//!
//! All parsers are built using the `nom` parser combinator library for
//! robust error handling and performance.

pub mod error;
pub mod touchstone;
pub mod ami_parser;
pub mod ibs_parser;

pub use error::ParseError;
pub use touchstone::{parse_touchstone, TouchstoneFile};
pub use ami_parser::{parse_ami_file, AmiFile};
pub use ibs_parser::{parse_ibs_file, IbisFile};
