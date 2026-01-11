//! AMI parameter file parser.
//!
//! Parses `.ami` files that define model parameters for IBIS-AMI models.
//! The format uses a Lisp-like S-expression syntax with keywords and values.
//!
//! Reference: IBIS-AMI Modeling Specification

use crate::error::ParseError;
use lib_types::ami::{AmiParameters, AmiValue};
use nom::{
    branch::alt,
    bytes::complete::{take_while, take_while1},
    character::complete::{char, multispace0},
    combinator::{map, opt, recognize},
    multi::many0,
    sequence::{pair, preceded, tuple},
    IResult, Parser,
};
use std::collections::HashMap;

/// Parsed AMI file.
#[derive(Clone, Debug)]
pub struct AmiFile {
    /// File name/identifier.
    pub name: String,

    /// Reserved parameters section.
    pub reserved_params: AmiParameters,

    /// Model-specific parameters section.
    pub model_specific: AmiParameters,

    /// Raw parameter tree for debugging.
    pub raw_tree: Option<SExpr>,
}

/// S-expression node for parsing.
#[derive(Clone, Debug)]
pub enum SExpr {
    /// Atom (identifier or string).
    Atom(String),
    /// Number.
    Number(f64),
    /// List of expressions.
    List(Vec<SExpr>),
}

impl SExpr {
    /// Try to get as a string atom.
    pub fn as_atom(&self) -> Option<&str> {
        match self {
            SExpr::Atom(s) => Some(s),
            _ => None,
        }
    }

    /// Try to get as a number.
    pub fn as_number(&self) -> Option<f64> {
        match self {
            SExpr::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// Try to get as a list.
    pub fn as_list(&self) -> Option<&[SExpr]> {
        match self {
            SExpr::List(l) => Some(l),
            _ => None,
        }
    }

    /// Find a named sub-expression.
    pub fn find(&self, name: &str) -> Option<&SExpr> {
        if let SExpr::List(items) = self {
            for item in items {
                if let SExpr::List(inner) = item {
                    if let Some(SExpr::Atom(first)) = inner.first() {
                        if first.eq_ignore_ascii_case(name) {
                            return Some(item);
                        }
                    }
                }
            }
        }
        None
    }

    /// Convert to AmiValue.
    pub fn to_ami_value(&self) -> AmiValue {
        match self {
            SExpr::Atom(s) => {
                if s.eq_ignore_ascii_case("true") {
                    AmiValue::Boolean(true)
                } else if s.eq_ignore_ascii_case("false") {
                    AmiValue::Boolean(false)
                } else if let Ok(i) = s.parse::<i64>() {
                    AmiValue::Integer(i)
                } else if let Ok(f) = s.parse::<f64>() {
                    AmiValue::Float(f)
                } else {
                    AmiValue::String(s.clone())
                }
            }
            SExpr::Number(n) => {
                if n.fract() == 0.0 && *n >= i64::MIN as f64 && *n <= i64::MAX as f64 {
                    AmiValue::Integer(*n as i64)
                } else {
                    AmiValue::Float(*n)
                }
            }
            SExpr::List(items) => {
                // Check if it's a list of numbers (tap values)
                let all_numbers = items.iter().all(|i| matches!(i, SExpr::Number(_)));
                if all_numbers {
                    let taps: Vec<f64> = items
                        .iter()
                        .filter_map(|i| i.as_number())
                        .collect();
                    AmiValue::Taps(taps)
                } else {
                    // General list
                    let values: Vec<AmiValue> = items.iter().map(|i| i.to_ami_value()).collect();
                    AmiValue::List(values)
                }
            }
        }
    }
}

/// Parse an AMI file from a string.
pub fn parse_ami_file(content: &str) -> Result<AmiFile, ParseError> {
    // First, strip comments
    let cleaned = strip_comments(content);

    // Parse as S-expression
    let (_remaining, expr) = parse_sexpr(&cleaned)
        .map_err(|e| ParseError::Nom(format!("{:?}", e)))?;

    // Extract structured data
    let file = extract_ami_file(&expr)?;

    Ok(file)
}

/// Strip comments (lines starting with | or text after |).
fn strip_comments(content: &str) -> String {
    content
        .lines()
        .map(|line| {
            if let Some(idx) = line.find('|') {
                &line[..idx]
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Extract AmiFile from parsed S-expression.
fn extract_ami_file(expr: &SExpr) -> Result<AmiFile, ParseError> {
    let list = expr.as_list().ok_or_else(|| {
        ParseError::InvalidFormat {
            format: "AMI".to_string(),
            message: "Root must be a list".to_string(),
        }
    })?;

    // Root should be (root_name ...)
    let name = list
        .first()
        .and_then(|e| e.as_atom())
        .unwrap_or("unknown")
        .to_string();

    let mut reserved_params = AmiParameters::new();
    let mut model_specific = AmiParameters::new();

    // Find Reserved_Parameters section
    if let Some(reserved) = expr.find("Reserved_Parameters") {
        extract_parameters(reserved, &mut reserved_params);
    }

    // Find Model_Specific section
    if let Some(model) = expr.find("Model_Specific") {
        extract_parameters(model, &mut model_specific);
    }

    Ok(AmiFile {
        name,
        reserved_params,
        model_specific,
        raw_tree: Some(expr.clone()),
    })
}

/// Extract parameters from a section.
fn extract_parameters(expr: &SExpr, params: &mut AmiParameters) {
    if let SExpr::List(items) = expr {
        // Skip the section name (first element)
        for item in items.iter().skip(1) {
            if let SExpr::List(pair) = item {
                if let Some(SExpr::Atom(name)) = pair.first() {
                    // Value is the rest of the list
                    if pair.len() == 2 {
                        let value = pair[1].to_ami_value();
                        params.set(name.clone(), value);
                    } else if pair.len() > 2 {
                        // Multiple values -> list
                        let values: Vec<AmiValue> = pair[1..]
                            .iter()
                            .map(|e| e.to_ami_value())
                            .collect();
                        params.set(name.clone(), AmiValue::List(values));
                    }
                }
            }
        }
    }
}

// ============================================================================
// S-expression Parser (nom 8 compatible)
// ============================================================================

fn parse_sexpr(input: &str) -> IResult<&str, SExpr> {
    let (input, _) = multispace0(input)?;
    alt((parse_list, parse_number, parse_string, parse_atom)).parse(input)
}

fn parse_list(input: &str) -> IResult<&str, SExpr> {
    let (input, _) = char('(')(input)?;
    let (input, items) = many0(preceded(multispace0, parse_sexpr)).parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')')(input)?;
    Ok((input, SExpr::List(items)))
}

fn parse_number(input: &str) -> IResult<&str, SExpr> {
    // Try to parse a number (including scientific notation)
    let (input, num_str) = recognize(pair(
        opt(char('-')),
        alt((
            // Scientific notation
            recognize(tuple((
                take_while1(|c: char| c.is_ascii_digit()),
                opt(pair(char('.'), take_while(|c: char| c.is_ascii_digit()))),
                pair(
                    alt((char('e'), char('E'))),
                    pair(opt(alt((char('+'), char('-')))), take_while1(|c: char| c.is_ascii_digit())),
                ),
            ))),
            // Regular decimal
            recognize(pair(
                take_while1(|c: char| c.is_ascii_digit()),
                opt(pair(char('.'), take_while(|c: char| c.is_ascii_digit()))),
            )),
        )),
    )).parse(input)?;

    let num: f64 = num_str.parse().map_err(|_| {
        nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Float))
    })?;

    Ok((input, SExpr::Number(num)))
}

fn parse_string(input: &str) -> IResult<&str, SExpr> {
    let (input, _) = char('"')(input)?;
    let (input, content) = take_while(|c| c != '"')(input)?;
    let (input, _) = char('"')(input)?;
    Ok((input, SExpr::Atom(content.to_string())))
}

fn parse_atom(input: &str) -> IResult<&str, SExpr> {
    let (input, atom) = take_while1(|c: char| {
        c.is_alphanumeric() || c == '_' || c == '-' || c == '.' || c == '/'
    })(input)?;

    Ok((input, SExpr::Atom(atom.to_string())))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_AMI: &str = r#"
(sample_ami_model
    (Reserved_Parameters
        (AMI_Version (Usage Info) (Type String) (Value "7.0"))
        (Init_Returns_Impulse (Usage Info) (Type Boolean) (Value True))
        (GetWave_Exists (Usage Info) (Type Boolean) (Value True))
    )
    (Model_Specific
        (Tx_Tap_Units (Usage Info) (Type String) (Value "dB"))
        (Tx_Tap_Main (Usage In) (Type Float) (Range 0.0 1.0) (Default 0.8))
        (Tx_Tap_Pre (Usage In) (Type Float) (Range -0.2 0.0) (Default -0.1))
    )
)
"#;

    #[test]
    fn test_parse_sample_ami() {
        let result = parse_ami_file(SAMPLE_AMI).unwrap();

        assert_eq!(result.name, "sample_ami_model");

        // Check reserved parameters
        assert!(result.reserved_params.contains("AMI_Version"));
        assert!(result.reserved_params.contains("Init_Returns_Impulse"));

        // Check model-specific parameters
        assert!(result.model_specific.contains("Tx_Tap_Units"));
        assert!(result.model_specific.contains("Tx_Tap_Main"));
    }

    #[test]
    fn test_parse_sexpr() {
        let input = "(test 1 2.5 \"hello\" (nested a b))";
        let (_, expr) = parse_sexpr(input).unwrap();

        if let SExpr::List(items) = expr {
            assert_eq!(items.len(), 5);
            assert_eq!(items[0].as_atom(), Some("test"));
            assert_eq!(items[1].as_number(), Some(1.0));
        } else {
            panic!("Expected list");
        }
    }
}
