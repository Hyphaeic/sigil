//! Result output formatting and writing.

use crate::orchestrator::SimulationResults;
use crate::OutputFormat;
use anyhow::Result;
use std::io::Write;
use std::path::Path;

/// Write simulation results to output directory.
pub fn write_results(results: &SimulationResults, output_dir: &Path, format: OutputFormat) -> Result<()> {
    // Write eye metrics
    if let Some(metrics) = &results.eye_metrics {
        let metrics_path = output_dir.join("eye_metrics.txt");
        let mut f = std::fs::File::create(&metrics_path)?;

        match format {
            OutputFormat::Text => {
                writeln!(f, "Eye Diagram Metrics")?;
                writeln!(f, "===================")?;
                writeln!(f, "Eye Height:     {:.6}", metrics.height)?;
                writeln!(f, "Eye Width (UI): {:.4}", metrics.width_ui)?;
                writeln!(f, "Jitter RMS:     {:.6}", metrics.jitter_rms)?;
                writeln!(f, "SNR:            {:.2} dB", metrics.snr)?;
                writeln!(f, "UI Count:       {}", metrics.ui_count)?;
            }
            OutputFormat::Json => {
                let json = serde_json::json!({
                    "eye_height": metrics.height,
                    "eye_width_ui": metrics.width_ui,
                    "jitter_rms": metrics.jitter_rms,
                    "snr": metrics.snr,
                    "ui_count": metrics.ui_count,
                });
                writeln!(f, "{}", serde_json::to_string_pretty(&json)?)?;
            }
            OutputFormat::Csv => {
                writeln!(f, "metric,value")?;
                writeln!(f, "eye_height,{}", metrics.height)?;
                writeln!(f, "eye_width_ui,{}", metrics.width_ui)?;
                writeln!(f, "jitter_rms,{}", metrics.jitter_rms)?;
                writeln!(f, "snr,{}", metrics.snr)?;
                writeln!(f, "ui_count,{}", metrics.ui_count)?;
            }
        }

        tracing::info!("Wrote eye metrics to {:?}", metrics_path);
    }

    // Write statistical eye data
    if let Some(stat_eye) = &results.statistical_eye {
        let stat_path = output_dir.join("statistical_eye.csv");
        let mut f = std::fs::File::create(&stat_path)?;

        writeln!(f, "phase,high,low,opening")?;
        for (i, (h, l)) in stat_eye.high.iter().zip(stat_eye.low.iter()).enumerate() {
            writeln!(f, "{},{},{},{}", i, h, l, h - l)?;
        }

        tracing::info!("Wrote statistical eye to {:?}", stat_path);
    }

    // Write pulse response
    if let Some(pulse) = &results.channel_pulse {
        let pulse_path = output_dir.join("pulse_response.csv");
        let mut f = std::fs::File::create(&pulse_path)?;

        writeln!(f, "time_ps,amplitude")?;
        for (i, &v) in pulse.samples.iter().enumerate() {
            let t = pulse.t_start.0 + i as f64 * pulse.dt.0;
            writeln!(f, "{},{}", t * 1e12, v)?;
        }

        tracing::info!("Wrote pulse response to {:?}", pulse_path);
    }

    // Write output waveform (sampled, to avoid huge files)
    if let Some(waveform) = &results.output_waveform {
        let wf_path = output_dir.join("output_waveform.csv");
        let mut f = std::fs::File::create(&wf_path)?;

        // Sample every Nth point if waveform is very long
        let max_points = 100_000;
        let step = (waveform.len() / max_points).max(1);

        writeln!(f, "sample,amplitude")?;
        for (i, &v) in waveform.samples.iter().enumerate().step_by(step) {
            writeln!(f, "{},{}", i, v)?;
        }

        tracing::info!(
            "Wrote output waveform to {:?} ({} of {} samples)",
            wf_path,
            waveform.len() / step,
            waveform.len()
        );
    }

    // Write summary
    let summary_path = output_dir.join("summary.txt");
    let mut f = std::fs::File::create(&summary_path)?;

    writeln!(f, "SI-Kernel Simulation Summary")?;
    writeln!(f, "============================")?;
    writeln!(f)?;

    if let Some(metrics) = &results.eye_metrics {
        writeln!(f, "Results:")?;
        writeln!(f, "  Eye Height: {:.4}", metrics.height)?;
        writeln!(f, "  Eye Width:  {:.2} UI", metrics.width_ui)?;

        // Simple pass/fail assessment
        let eye_open = metrics.height > 0.0 && metrics.width_ui > 0.3;
        writeln!(f)?;
        if eye_open {
            writeln!(f, "Status: PASS - Eye is open")?;
        } else {
            writeln!(f, "Status: FAIL - Eye is closed or marginal")?;
        }
    }

    if let Some(training) = &results.training_result {
        writeln!(f)?;
        writeln!(f, "Training Results:")?;
        writeln!(f, "  Best Preset: P{}", training.best_preset)?;
        writeln!(f, "  Final FOM:   {:.4}", training.final_fom)?;
        writeln!(f, "  Iterations:  {}", training.iterations)?;
    }

    tracing::info!("Wrote summary to {:?}", summary_path);

    Ok(())
}

/// Print results to stdout.
pub fn print_results(results: &SimulationResults) {
    println!("\n=== Simulation Results ===\n");

    if let Some(metrics) = &results.eye_metrics {
        println!("Eye Metrics:");
        println!("  Height:     {:.6}", metrics.height);
        println!("  Width (UI): {:.4}", metrics.width_ui);

        if metrics.height > 0.0 && metrics.width_ui > 0.3 {
            println!("\n  Status: PASS");
        } else {
            println!("\n  Status: FAIL");
        }
    }

    if let Some(training) = &results.training_result {
        println!("\nTraining:");
        println!("  Best Preset: P{}", training.best_preset);
        println!("  Final FOM:   {:.4}", training.final_fom);
    }

    println!();
}
