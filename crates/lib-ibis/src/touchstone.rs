//! Touchstone S-parameter file parser.
//!
//! Supports:
//! - Touchstone 1.0 format (.s1p, .s2p, .s4p, .s6p, .s8p, etc.)
//! - All data formats: RI, MA, DB
//! - Frequency units: Hz, kHz, MHz, GHz
//!
//! Reference: Touchstone File Format Specification, Version 1.1

use crate::error::ParseError;
use lib_types::{
    sparams::{DataFormat, SMatrix, SParameters, TouchstoneVersion},
    units::{Hertz, Ohms},
    Complex64,
};
use ndarray::Array2;
use nom::{
    branch::alt,
    bytes::complete::{tag_no_case, take_while1},
    character::complete::{char, line_ending, not_line_ending, space0, space1},
    combinator::{map, opt, value},
    multi::{many0, many1},
    number::complete::double,
    sequence::preceded,
    IResult, Parser,
};
use std::path::Path;

/// Parsed Touchstone file.
#[derive(Clone, Debug)]
pub struct TouchstoneFile {
    /// File version.
    pub version: TouchstoneVersion,

    /// Number of ports.
    pub num_ports: usize,

    /// Data format (RI, MA, DB).
    pub format: DataFormat,

    /// Reference impedance.
    pub z0: Ohms,

    /// Frequency multiplier (Hz, kHz, MHz, GHz).
    pub freq_mult: f64,

    /// Parsed S-parameters.
    pub sparams: SParameters,
}

impl TouchstoneFile {
    /// Get the S-parameters.
    pub fn into_sparams(self) -> SParameters {
        self.sparams
    }
}

/// Parse a Touchstone file from a string.
pub fn parse_touchstone(content: &str) -> Result<TouchstoneFile, ParseError> {
    parse_touchstone_with_ports(content, None)
}

/// Parse Touchstone content with a known port count.
///
/// Prefer this (or [`parse_touchstone_file`], which derives the count from
/// the `.sNp` extension) over [`parse_touchstone`] whenever the port count
/// is known: line-shape inference cannot always disambiguate — a wrapped
/// 4-port file whose rows span lines of 9/8/8/8 values is indistinguishable
/// from single-line 2-port data when the total happens to divide evenly.
pub fn parse_touchstone_with_ports(
    content: &str,
    ports: Option<usize>,
) -> Result<TouchstoneFile, ParseError> {
    let (remaining, file) = parse_touchstone_inner(content, ports)
        .map_err(|e| ParseError::Nom(format!("{:?}", e)))?;

    if !remaining.trim().is_empty() {
        tracing::warn!("Unparsed content at end of Touchstone file");
    }

    Ok(file)
}

/// Parse a Touchstone file from a path.
pub fn parse_touchstone_file(path: &Path) -> Result<TouchstoneFile, ParseError> {
    let content = std::fs::read_to_string(path)?;

    // The .sNp extension is the authoritative port count; data-shape
    // inference is only a fallback for extension-less content.
    let num_ports = infer_ports_from_extension(path);

    let file = parse_touchstone_with_ports(&content, num_ports)?;

    Ok(file)
}

fn infer_ports_from_extension(path: &Path) -> Option<usize> {
    let ext = path.extension()?.to_str()?;
    if ext.starts_with('s') || ext.starts_with('S') {
        let num_part = &ext[1..ext.len() - 1];
        num_part.parse().ok()
    } else {
        None
    }
}

// ============================================================================
// Nom Parsers (nom 8 compatible)
// ============================================================================

fn parse_touchstone_inner(
    input: &str,
    ports_hint: Option<usize>,
) -> IResult<&str, TouchstoneFile> {
    let (input, _) = many0(comment_or_blank_line).parse(input)?;
    let (input, options) = parse_options_line(input)?;
    // Comments and blank lines may appear anywhere inside the data section,
    // not just before it — VNA/IdEM exports commonly separate every
    // frequency point with a bare `!` line.
    let (input, data_lines) =
        many1(preceded(many0(comment_or_blank_line), parse_data_line)).parse(input)?;
    let (input, _) = many0(comment_or_blank_line).parse(input)?;

    // The file extension (.sNp) is authoritative when available; line-shape
    // inference cannot always disambiguate (e.g. a wrapped 4-port file whose
    // first data line holds 9 values looks exactly like single-line 2-port).
    let num_ports = match ports_hint {
        Some(n) => n,
        None => infer_ports_from_data(&data_lines).map_err(|_| {
            nom::Err::Failure(nom::error::Error::new(input, nom::error::ErrorKind::Verify))
        })?,
    };

    // Build S-parameters
    let sparams = build_sparams(&options, &data_lines, num_ports).map_err(|_| {
        nom::Err::Failure(nom::error::Error::new(input, nom::error::ErrorKind::Verify))
    })?;

    Ok((
        input,
        TouchstoneFile {
            version: TouchstoneVersion::V1,
            num_ports,
            format: options.format,
            z0: options.z0,
            freq_mult: options.freq_mult,
            sparams,
        },
    ))
}

/// Options from the # line.
#[derive(Clone, Debug)]
struct OptionsLine {
    freq_mult: f64,
    param_type: char,  // S, Y, Z, H, G
    format: DataFormat,
    z0: Ohms,
}

impl Default for OptionsLine {
    fn default() -> Self {
        Self {
            freq_mult: 1e9,  // GHz default
            param_type: 'S',
            format: DataFormat::MA,
            z0: Ohms::Z0_50,
        }
    }
}

fn parse_options_line(input: &str) -> IResult<&str, OptionsLine> {
    let (input, _) = space0(input)?;
    let (input, _) = char('#')(input)?;
    let (input, _) = space0(input)?;

    let mut options = OptionsLine::default();

    // Parse tokens in any order
    let (input, tokens) = many0(preceded(space0, parse_option_token)).parse(input)?;

    for token in tokens {
        match token {
            OptionToken::FreqUnit(mult) => options.freq_mult = mult,
            OptionToken::ParamType(t) => options.param_type = t,
            OptionToken::Format(f) => options.format = f,
            OptionToken::Z0(z) => options.z0 = z,
        }
    }

    let (input, _) = opt(preceded(space0, not_line_ending)).parse(input)?;
    let (input, _) = opt(line_ending).parse(input)?;

    Ok((input, options))
}

#[derive(Clone, Debug)]
enum OptionToken {
    FreqUnit(f64),
    ParamType(char),
    Format(DataFormat),
    Z0(Ohms),
}

fn parse_option_token(input: &str) -> IResult<&str, OptionToken> {
    alt((
        parse_freq_unit,
        parse_param_type,
        parse_format,
        parse_z0,
    )).parse(input)
}

fn parse_freq_unit(input: &str) -> IResult<&str, OptionToken> {
    alt((
        value(OptionToken::FreqUnit(1.0), tag_no_case("HZ")),
        value(OptionToken::FreqUnit(1e3), tag_no_case("KHZ")),
        value(OptionToken::FreqUnit(1e6), tag_no_case("MHZ")),
        value(OptionToken::FreqUnit(1e9), tag_no_case("GHZ")),
    )).parse(input)
}

fn parse_param_type(input: &str) -> IResult<&str, OptionToken> {
    alt((
        value(OptionToken::ParamType('S'), tag_no_case("S")),
        value(OptionToken::ParamType('Y'), tag_no_case("Y")),
        value(OptionToken::ParamType('Z'), tag_no_case("Z")),
        value(OptionToken::ParamType('H'), tag_no_case("H")),
        value(OptionToken::ParamType('G'), tag_no_case("G")),
    )).parse(input)
}

fn parse_format(input: &str) -> IResult<&str, OptionToken> {
    alt((
        value(OptionToken::Format(DataFormat::RI), tag_no_case("RI")),
        value(OptionToken::Format(DataFormat::MA), tag_no_case("MA")),
        value(OptionToken::Format(DataFormat::DB), tag_no_case("DB")),
    )).parse(input)
}

fn parse_z0(input: &str) -> IResult<&str, OptionToken> {
    let (input, _) = tag_no_case("R")(input)?;
    let (input, _) = space1(input)?;
    let (input, z0) = double(input)?;
    Ok((input, OptionToken::Z0(Ohms(z0))))
}

fn comment_or_blank_line(input: &str) -> IResult<&str, ()> {
    alt((
        map(
            (space0, char('!'), not_line_ending, opt(line_ending)),
            |_| (),
        ),
        map((space0, line_ending), |_| ()),
    )).parse(input)
}

fn parse_data_line(input: &str) -> IResult<&str, Vec<f64>> {
    let (input, _) = space0(input)?;

    // Skip if it's a comment or empty
    if input.starts_with('!') || input.starts_with('\n') || input.starts_with('\r') || input.is_empty() {
        return Err(nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Tag)));
    }

    let (input, values) = many1(preceded(space0, double)).parse(input)?;
    let (input, _) = opt(preceded(space0, preceded(char('!'), not_line_ending))).parse(input)?;
    let (input, _) = opt(line_ending).parse(input)?;

    Ok((input, values))
}

/// Infer number of ports from data.
///
/// Returns Ok(num_ports) if successfully inferred, Err if ambiguous or unsupported.
///
/// # Port count determination
///
/// For N ports, each frequency has N² S-parameters (2 values each for complex)
/// Plus 1 frequency value = 1 + 2*N² values per frequency point:
/// - 1-port: 1 + 2*1 = 3 values
/// - 2-port: 1 + 2*4 = 9 values
/// - 3-port: 1 + 2*9 = 19 values
/// - 4-port: 1 + 2*16 = 33 values
/// - 6-port: 1 + 2*36 = 73 values
/// - 8-port: 1 + 2*64 = 129 values
fn infer_ports_from_data(data_lines: &[Vec<f64>]) -> Result<usize, ParseError> {
    if data_lines.is_empty() {
        return Err(ParseError::InvalidFormat {
            format: "Touchstone".to_string(),
            message: "No data lines found in file".to_string(),
        });
    }

    // Count total values across all lines
    let total_values: usize = data_lines.iter().map(|line| line.len()).sum();

    // Try to determine port count from total values
    // We need total_values to be a multiple of (1 + 2*N²)

    let first_line_len = data_lines[0].len();

    // For 1-port and 2-port, prioritize based on first line structure
    // 1-port: 3 values per line (freq, S11_re, S11_im)
    // 2-port: 9 values per line (freq, S11, S21, S12, S22 - each 2 values)

    // Wrapped 4-port (VNA/IdEM exports): each frequency point spans lines of
    // 9, 8, 8, 8 values (freq + matrix row 1, then rows 2-4). The first line
    // alone is indistinguishable from single-line 2-port, so check the
    // continuation-line signature before the 2-port shortcut.
    if first_line_len == 9
        && data_lines.len() >= 4
        && data_lines[1].len() == 8
        && total_values % 33 == 0
    {
        return Ok(4);
    }

    // Check exact matches first to avoid ambiguity
    if first_line_len == 9 {
        // Definitely 2-port (all data on one line)
        let values_per_freq = 1 + 2 * 2 * 2; // = 9
        if total_values % values_per_freq == 0 {
            return Ok(2);
        }
    }

    if first_line_len == 3 {
        // Definitely 1-port
        let values_per_freq = 1 + 2 * 1 * 1; // = 3
        if total_values % values_per_freq == 0 {
            return Ok(1);
        }
    }

    // For multi-line formats or larger ports, check modulo
    for num_ports in [2, 3, 4, 6, 8, 1] {  // Check 2-port before 1-port to prefer larger
        let values_per_freq = 1 + 2 * num_ports * num_ports;
        if total_values % values_per_freq == 0 && total_values >= values_per_freq {
            // For larger ports (3+), data typically spans multiple lines
            if num_ports >= 3 {
                return Ok(num_ports);
            }
            // For 1-port and 2-port multi-line formats
            if num_ports == 2 && first_line_len >= 5 {
                return Ok(2);
            }
            if num_ports == 1 && first_line_len >= 3 {
                return Ok(1);
            }
        }
    }

    // If we can't determine, check first line as fallback
    match first_line_len {
        3 => Ok(1),
        9 => Ok(2),
        5 => Ok(2), // Sometimes 2-port on multiple lines
        n => Err(ParseError::InvalidFormat {
            format: "Touchstone".to_string(),
            message: format!(
                "Cannot infer port count from {} values per line. \
                Expected 3 (1-port), 9 (2-port), or multi-line format for larger ports.",
                n
            ),
        }),
    }
}

fn build_sparams(
    options: &OptionsLine,
    data_lines: &[Vec<f64>],
    num_ports: usize,
) -> Result<SParameters, ParseError> {
    let mut sparams = SParameters::new(num_ports, options.z0);

    // Flatten all data
    let all_values: Vec<f64> = data_lines.iter().flatten().copied().collect();

    // Values per frequency point: 1 (freq) + 2 * N² (complex S-params)
    let values_per_freq = 1 + 2 * num_ports * num_ports;

    if all_values.len() < values_per_freq {
        return Err(ParseError::InvalidFormat {
            format: "Touchstone".to_string(),
            message: format!(
                "Insufficient data: need at least {} values for {}-port, got {}",
                values_per_freq, num_ports, all_values.len()
            ),
        });
    }

    // Parse frequency points using chunks for safety
    for chunk in all_values.chunks_exact(values_per_freq) {
        let freq = Hertz(chunk[0] * options.freq_mult);
        let params = &chunk[1..];

        // Verify we have the right number of parameters
        if params.len() != 2 * num_ports * num_ports {
            return Err(ParseError::InvalidFormat {
                format: "Touchstone".to_string(),
                message: format!(
                    "Expected {} S-parameter values, got {}",
                    2 * num_ports * num_ports,
                    params.len()
                ),
            });
        }

        let mut matrix = Array2::zeros((num_ports, num_ports));

        if num_ports == 2 {
            // Touchstone 1.0 quirk: 2-port data order is S11 S21 S12 S22
            // (NOT row-major). Using row-major here silently swapped
            // S21/S12 — invisible for reciprocal channels, wrong per spec.
            let order = [(0, 0), (1, 0), (0, 1), (1, 1)];
            for (i, &(row, col)) in order.iter().enumerate() {
                matrix[[row, col]] =
                    options.format.to_complex(params[2 * i], params[2 * i + 1]);
            }
        } else {
            // 1-port and 3+ ports: row-major order per spec.
            let mut param_idx = 0;
            for row in 0..num_ports {
                for col in 0..num_ports {
                    let val1 = params[param_idx];
                    let val2 = params[param_idx + 1];
                    param_idx += 2;

                    matrix[[row, col]] = options.format.to_complex(val1, val2);
                }
            }
        }

        sparams.add_point(freq, matrix);
    }

    // Warn if there are leftover values (incomplete frequency point)
    let remainder = all_values.len() % values_per_freq;
    if remainder != 0 {
        tracing::warn!(
            "Incomplete frequency point: {} extra values ignored",
            remainder
        );
    }

    Ok(sparams)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_S2P: &str = r#"! Sample 2-port S-parameter file
# GHz S RI R 50
! freq  S11_re S11_im S21_re S21_im S12_re S12_im S22_re S22_im
1.0  0.1 0.0  0.9 0.0  0.9 0.0  0.1 0.0
2.0  0.15 0.05  0.85 -0.1  0.85 -0.1  0.15 0.05
"#;

    /// Wrapped 4-port export in the style of real VNA/IdEM files from the
    /// IEEE 802.3ck channel repository: each frequency point spans lines of
    /// 9/8/8/8 values, points separated by bare `!` comment lines, and the
    /// sweep starts at DC. Shape inference alone cannot distinguish the
    /// 9-value first line from single-line 2-port data.
    const WRAPPED_S4P: &str = r#"! IdEM-style wrapped 4-port export
# Hz S RI R 50
!
0  0.1 0.0  0.9 0.0  0.0 0.0  0.0 0.0
   0.9 0.0  0.1 0.0  0.0 0.0  0.0 0.0
   0.0 0.0  0.0 0.0  0.1 0.0  0.9 0.0
   0.0 0.0  0.0 0.0  0.9 0.0  0.1 0.0
!
1000000000  0.12 0.01  0.8 -0.2  0.0 0.0  0.0 0.0
   0.8 -0.2  0.12 0.01  0.0 0.0  0.0 0.0
   0.0 0.0  0.0 0.0  0.12 0.01  0.8 -0.2
   0.0 0.0  0.0 0.0  0.8 -0.2  0.12 0.01
"#;

    #[test]
    fn test_parse_wrapped_4port_with_interleaved_comments() {
        // Extension hint path (authoritative).
        let hinted = parse_touchstone_with_ports(WRAPPED_S4P, Some(4)).unwrap();
        assert_eq!(hinted.num_ports, 4);
        assert_eq!(hinted.sparams.len(), 2);

        // Inference path: 9/8/8/8 continuation signature beats the
        // "9 values = 2-port" shortcut.
        let inferred = parse_touchstone(WRAPPED_S4P).unwrap();
        assert_eq!(inferred.num_ports, 4);
        assert_eq!(inferred.sparams.len(), 2);

        // DC point preserved; row-major S21 (= matrix[1][0]) is the through.
        assert_eq!(hinted.sparams.frequencies[0].0, 0.0);
        let s21 = hinted.sparams.get_parameter(1, 0);
        assert!((s21[0].re - 0.9).abs() < 1e-12);
        assert!((s21[1].re - 0.8).abs() < 1e-12 && (s21[1].im + 0.2).abs() < 1e-12);
    }

    /// Touchstone 1.0 quirk: 2-port column order is S11 S21 S12 S22.
    /// Asymmetric data distinguishes correct ordering from row-major.
    #[test]
    fn test_2port_spec_column_order() {
        let asym = "! asymmetric 2-port\n# GHz S RI R 50\n1.0  0.1 0.0  0.9 0.0  0.5 0.0  0.2 0.0\n";
        let ts = parse_touchstone(asym).unwrap();
        let s21 = ts.sparams.get_parameter(1, 0)[0];
        let s12 = ts.sparams.get_parameter(0, 1)[0];
        assert!((s21.re - 0.9).abs() < 1e-12, "second pair is S21 per spec");
        assert!((s12.re - 0.5).abs() < 1e-12, "third pair is S12 per spec");
    }

    #[test]
    fn test_parse_sample_s2p() {
        let result = parse_touchstone(SAMPLE_S2P).unwrap();

        assert_eq!(result.num_ports, 2);
        assert_eq!(result.format, DataFormat::RI);
        assert_eq!(result.z0.0, 50.0);
        assert_eq!(result.sparams.len(), 2);

        // Check first frequency point
        let freqs = &result.sparams.frequencies;
        assert!((freqs[0].0 - 1e9).abs() < 1.0);
        assert!((freqs[1].0 - 2e9).abs() < 1.0);

        // Check S21 at first point
        let s21 = result.sparams.s21();
        assert!((s21[0].re - 0.9).abs() < 1e-10);
        assert!((s21[0].im - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_options_parsing() {
        let input = "# MHZ S DB R 75\n";
        let (_, options) = parse_options_line(input).unwrap();

        assert!((options.freq_mult - 1e6).abs() < 1.0);
        assert_eq!(options.param_type, 'S');
        assert_eq!(options.format, DataFormat::DB);
        assert!((options.z0.0 - 75.0).abs() < 1e-10);
    }
}
