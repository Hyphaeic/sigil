//! Convolution performance benchmarks.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use lib_dsp::convolution::{direct_convolve, fft_convolve, ConvolutionEngine};

fn bench_convolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("convolution");

    // Different signal lengths
    for signal_len in [1024, 4096, 16384, 65536].iter() {
        let signal: Vec<f64> = (0..*signal_len).map(|i| (i as f64 * 0.01).sin()).collect();
        let kernel: Vec<f64> = (0..256).map(|i| (-i as f64 * 0.1).exp()).collect();

        // Only benchmark direct convolution for small sizes
        if *signal_len <= 4096 {
            group.bench_with_input(
                BenchmarkId::new("direct", signal_len),
                &(&signal, &kernel),
                |b, (s, k)| {
                    b.iter(|| direct_convolve(black_box(s), black_box(k)));
                },
            );
        }

        group.bench_with_input(
            BenchmarkId::new("fft_single", signal_len),
            &(&signal, &kernel),
            |b, (s, k)| {
                b.iter(|| fft_convolve(black_box(s), black_box(k)));
            },
        );

        let engine = ConvolutionEngine::new(&kernel).unwrap();
        group.bench_with_input(
            BenchmarkId::new("engine", signal_len),
            &(&signal, &engine),
            |b, (s, e)| {
                b.iter(|| e.convolve(black_box(s)));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_convolution);
criterion_main!(benches);
