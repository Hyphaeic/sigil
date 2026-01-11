//! SI-Kernel CLI: High-speed signal integrity simulation for PCIe Gen 5/6.
//!
//! This is the main entry point for the SI-Kernel simulation tool.

mod config;
mod orchestrator;
mod output;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

#[derive(Parser)]
#[command(name = "si-kernel")]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Enable verbose output
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Output format
    #[arg(short, long, default_value = "text")]
    format: OutputFormat,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
enum OutputFormat {
    #[default]
    Text,
    Json,
    Csv,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a channel simulation
    Simulate {
        /// Path to the simulation configuration file
        #[arg(short, long)]
        config: PathBuf,

        /// Output directory for results
        #[arg(short, long, default_value = "output")]
        output: PathBuf,
    },

    /// Parse and validate an IBIS file
    ParseIbis {
        /// Path to the .ibs file
        file: PathBuf,
    },

    /// Parse and validate a Touchstone file
    ParseTouchstone {
        /// Path to the .sNp file
        file: PathBuf,

        /// Convert to pulse response
        #[arg(long)]
        to_pulse: bool,

        /// Bit time for pulse response (ps)
        #[arg(long, default_value = "31.25")]
        bit_time_ps: f64,
    },

    /// Parse and validate an AMI file
    ParseAmi {
        /// Path to the .ami file
        file: PathBuf,
    },

    /// Generate a PRBS waveform
    GeneratePrbs {
        /// PRBS order (7, 9, 11, 15, 23, 31)
        #[arg(short, long, default_value = "31")]
        order: u8,

        /// Number of bits to generate
        #[arg(short, long, default_value = "1000000")]
        bits: u64,

        /// Samples per bit
        #[arg(short, long, default_value = "64")]
        samples_per_bit: usize,

        /// Output file path
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Compute eye diagram from a waveform
    Eye {
        /// Path to waveform file (CSV or binary)
        waveform: PathBuf,

        /// Samples per UI
        #[arg(long, default_value = "64")]
        samples_per_ui: usize,

        /// Output image path
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let filter = match cli.verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };

    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(filter)))
        .init();

    match cli.command {
        Commands::Simulate { config, output } => {
            run_simulation(&config, &output, cli.format)?;
        }
        Commands::ParseIbis { file } => {
            parse_ibis(&file)?;
        }
        Commands::ParseTouchstone { file, to_pulse, bit_time_ps } => {
            parse_touchstone(&file, to_pulse, bit_time_ps)?;
        }
        Commands::ParseAmi { file } => {
            parse_ami(&file)?;
        }
        Commands::GeneratePrbs { order, bits, samples_per_bit, output } => {
            generate_prbs(order, bits, samples_per_bit, output)?;
        }
        Commands::Eye { waveform, samples_per_ui, output } => {
            compute_eye(&waveform, samples_per_ui, output)?;
        }
    }

    Ok(())
}

fn run_simulation(config_path: &PathBuf, output_dir: &PathBuf, format: OutputFormat) -> Result<()> {
    tracing::info!("Loading configuration from {:?}", config_path);

    let config = config::load_config(config_path)?;
    let orchestrator = orchestrator::Orchestrator::new(config)?;

    tracing::info!("Starting simulation...");
    let results = orchestrator.run()?;

    // Create output directory
    std::fs::create_dir_all(output_dir)?;

    // Write results
    output::write_results(&results, output_dir, format)?;

    tracing::info!("Simulation complete. Results written to {:?}", output_dir);
    Ok(())
}

fn parse_ibis(file: &PathBuf) -> Result<()> {
    tracing::info!("Parsing IBIS file: {:?}", file);

    let content = std::fs::read_to_string(file)?;
    let ibis = lib_ibis::parse_ibs_file(&content)?;

    println!("IBIS File: {}", ibis.header.file_name);
    println!("Version: {}", ibis.header.ibis_ver);
    println!("Source: {}", ibis.header.source);
    println!("Components: {}", ibis.components.len());
    println!("Models: {}", ibis.models.len());

    for component in &ibis.components {
        println!("\n  Component: {}", component.name);
        println!("    Pins: {}", component.pins.len());
    }

    for model in &ibis.models {
        println!("\n  Model: {}", model.name);
        println!("    Type: {:?}", model.model_type);
        if model.algorithmic_model.is_some() {
            println!("    Has Algorithmic Model: Yes");
        }
    }

    Ok(())
}

fn parse_touchstone(file: &PathBuf, to_pulse: bool, bit_time_ps: f64) -> Result<()> {
    tracing::info!("Parsing Touchstone file: {:?}", file);

    let content = std::fs::read_to_string(file)?;
    let ts = lib_ibis::parse_touchstone(&content)?;

    println!("Touchstone File");
    println!("  Ports: {}", ts.num_ports);
    println!("  Format: {:?}", ts.format);
    println!("  Z0: {} ohms", ts.z0.0);
    println!("  Frequency points: {}", ts.sparams.len());

    if let Some((f_min, f_max)) = ts.sparams.frequency_range() {
        println!("  Frequency range: {:.2} MHz - {:.2} GHz",
            f_min.as_mhz(), f_max.as_ghz());
    }

    // Check passivity
    if ts.sparams.is_passive() {
        println!("  Passive: Yes");
    } else {
        println!("  Passive: No (may require enforcement)");
    }

    if to_pulse {
        use lib_dsp::sparam_convert::{sparam_to_pulse, ConversionConfig};
        use lib_types::units::Seconds;

        let config = ConversionConfig {
            bit_time: Seconds::from_ps(bit_time_ps),
            ..Default::default()
        };

        match sparam_to_pulse(&ts.sparams, &config) {
            Ok(pulse) => {
                println!("\nPulse Response Generated:");
                println!("  Samples: {}", pulse.len());
                println!("  Duration: {:.2} ns", pulse.duration().as_ns());
                println!("  Peak: {:.4}", pulse.max_abs());
            }
            Err(e) => {
                println!("\nFailed to generate pulse response: {}", e);
            }
        }
    }

    Ok(())
}

fn parse_ami(file: &PathBuf) -> Result<()> {
    tracing::info!("Parsing AMI file: {:?}", file);

    let content = std::fs::read_to_string(file)?;
    let ami = lib_ibis::parse_ami_file(&content)?;

    println!("AMI File: {}", ami.name);
    println!("\nReserved Parameters:");
    for (name, value) in &ami.reserved_params.params {
        println!("  {}: {:?}", name, value);
    }

    println!("\nModel-Specific Parameters:");
    for (name, value) in &ami.model_specific.params {
        println!("  {}: {:?}", name, value);
    }

    Ok(())
}

fn generate_prbs(order: u8, bits: u64, samples_per_bit: usize, output: Option<PathBuf>) -> Result<()> {
    use lib_dsp::prbs::PrbsGenerator;
    use lib_types::units::Seconds;

    tracing::info!("Generating PRBS-{} with {} bits", order, bits);

    let mut gen = PrbsGenerator::new(order);
    let waveform = gen.generate_nrz(bits, samples_per_bit, Seconds::from_ps(1.0));

    println!("Generated PRBS-{} waveform:", order);
    println!("  Bits: {}", bits);
    println!("  Samples: {}", waveform.len());
    println!("  Samples/bit: {}", samples_per_bit);

    if let Some(output_path) = output {
        // Write as CSV
        let mut writer = std::fs::File::create(&output_path)?;
        use std::io::Write;
        writeln!(writer, "sample,value")?;
        for (i, &v) in waveform.samples.iter().enumerate() {
            writeln!(writer, "{},{}", i, v)?;
        }
        println!("  Written to: {:?}", output_path);
    }

    Ok(())
}

fn compute_eye(waveform_path: &PathBuf, samples_per_ui: usize, output: Option<PathBuf>) -> Result<()> {
    tracing::info!("Computing eye diagram from {:?}", waveform_path);

    // For now, just print a placeholder
    println!("Eye diagram computation not yet implemented.");
    println!("Would analyze: {:?}", waveform_path);
    println!("Samples per UI: {}", samples_per_ui);
    if let Some(out) = output {
        println!("Output would go to: {:?}", out);
    }

    Ok(())
}
