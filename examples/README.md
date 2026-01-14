# SI-Kernel Examples

This directory contains example configurations and test files for the SI-Kernel simulation framework.

## Quick Start: Test All Fixes Now! ✅

```bash
# Run the channel-only test (no vendor files needed)
./test_crit_phys_fixes.sh
```

**✅ Updated Jan 13, 2026** — This validates all critical fixes:

**Physics Fixes (CRIT-PHYS):**
- **CRIT-PHYS-001:** DC extrapolation enforces S21(0) = 1.0
- **CRIT-PHYS-002:** FFT grid starts at 0 Hz (preserves Hilbert pairs)
- **CRIT-PHYS-003:** Impulse integration includes dt scaling

**Jan 13 Remediation Fixes:**
- **CRIT-NEW-001:** AMI_Free memory leak prevention (ready for vendor models)
- **CRIT-NEW-002:** Bit-by-bit transient discard (eye diagrams now correct)
- **HIGH-NEW-003:** Statistical sampling auto-detection (IBIS compliant)
- **HIGH-NEW-004:** Link training validation (clear errors, no silent failures)

## Files in This Directory

### Test Files (Ready to Use)

**Single-Ended Channel:**
- **`test_channel.s2p`** - Synthetic 2-port PCIe channel model
  - Represents ~10 inch PCB trace
  - Frequency range: 100 MHz - 20 GHz
  - Typical FR-4 dielectric loss
- **`channel_only_test.json`** - Single-ended configuration
  - Channel-only simulation (no TX/RX models)
  - Statistical eye diagram mode
  - 100,000 bits @ PCIe Gen 5 (32 GT/s)

**Differential Channel (Track B):**
- **`test_channel_4port.s4p`** - Synthetic 4-port differential channel
  - Same electrical parameters as 2-port
  - Supports mixed-mode analysis (SDD, SDC, SCD)
- **`differential_test.json`** - Differential configuration
  - Uses IEEE P370-2020 mixed-mode conversion
  - Reports mode conversion ratios
  - Validates HIGH-PHYS-004 fix

**Validation Scripts:**
- **`test_crit_phys_fixes.sh`** - Automated test script
  - Builds project
  - Verifies test files
  - Runs simulation and unit tests
  - Validates all Jan 13 fixes

### Example Configurations
- **`pcie_gen5_config.json`** - Full PCIe Gen 5 simulation
  - Requires vendor IBIS/AMI models
  - Includes TX/RX equalization
  - Link training enabled

- **`simple_channel.rs`** - Rust example code
  - Shows API usage
  - Programmatic configuration

### Documentation
- **`VENDOR_FILES_GUIDE.md`** - Complete guide for obtaining vendor files
  - Where to get IBIS models
  - How to register with Intel/AMD
  - NDA requirements for binary models

## Testing Levels

| Level | Files Needed | What You Can Test | Status |
|-------|--------------|-------------------|--------|
| **1. Channel Only** | `.sNp` Touchstone | S-param conversion, eye diagrams | ✅ **Ready Now** |
| **2. + IBIS Buffers** | `.ibs` files | I/O buffer characteristics | 🟡 Register with vendors |
| **3. + AMI Params** | `.ami` files | EQ parameters, no binary models | 🟡 Download from vendors |
| **4. + AMI Binaries** | `.so/.dll` files | Full TX/RX equalization | 🔴 NDA often required |

## Usage Examples

### 1. Channel-Only Simulation (No Vendor Files)
```bash
# Use provided synthetic channel
./test_crit_phys_fixes.sh
```

### 2. Custom Channel File
```bash
# Replace test_channel.s2p with your VNA measurements
cp /path/to/my_channel.s4p .

# Update config
vim channel_only_test.json
# Change: "touchstone": "my_channel.s4p"

# Run simulation
cargo run --release -- simulate channel_only_test.json
```

### 3. Full TX/RX Simulation (Requires Vendor Files)
```bash
# 1. Obtain files (see VENDOR_FILES_GUIDE.md)
# 2. Place in examples/ directory:
#    - tx_buffer.ibs
#    - rx_buffer.ibs
#    - tx_model.ami
#    - rx_model.ami
#    - tx_model.so (or .dll on Windows)
#    - rx_model.so

# 3. Update pcie_gen5_config.json paths

# 4. Run full simulation
cargo run --release -- simulate pcie_gen5_config.json
```

## What Gets Tested

### Channel-Only Mode ✅
- [x] Touchstone file parsing
- [x] S-parameter interpolation
- [x] DC extrapolation (S21(0) = 1.0)
- [x] FFT grid construction starting at 0 Hz
- [x] Causality enforcement with group delay preservation
- [x] Passivity validation via SVD
- [x] Impulse-to-pulse conversion with dt scaling
- [x] PRBS pattern generation
- [x] Convolution engine
- [x] Statistical eye diagram analysis
- [x] Bit-by-bit simulation

### With IBIS Models (Level 2)
- [ ] IBIS file parsing
- [ ] I/O buffer I-V curve interpolation
- [ ] Rising/falling edge waveform generation
- [ ] Driver impedance effects
- [ ] Load capacitance modeling

### With AMI Parameters (Level 3)
- [ ] AMI parameter file parsing
- [ ] TX preset configuration
- [ ] RX CTLE/DFE parameters
- [ ] Figure-of-merit calculation
- [ ] Link training state machine

### With AMI Binaries (Level 4)
- [ ] AMI_Init lifecycle
- [ ] AMI_GetWave processing
- [ ] AMI_Close cleanup
- [ ] Back-channel message passing
- [ ] Adaptive equalization
- [ ] Full TX-RX co-simulation

## Common Issues

### "File not found" errors
```bash
# Make sure you're in the examples/ directory
cd /path/to/si-kernel/examples

# Or use absolute paths in config files
vim channel_only_test.json
# "touchstone": "/absolute/path/to/test_channel.s2p"
```

### Parser errors with Touchstone files
```bash
# Check file format
head -5 test_channel.s2p

# Should start with:
# ! comment lines
# # GHz S MA R 50
```

### Missing vendor files
See `VENDOR_FILES_GUIDE.md` for:
- Where to download IBIS models
- How to register with vendors
- Alternative: synthesize your own AMI models

## Next Steps

1. **Test Now** ✅
   ```bash
   ./test_crit_phys_fixes.sh
   ```

2. **Get Vendor Files** (Week 1-2)
   - Read `VENDOR_FILES_GUIDE.md`
   - Register with Intel/AMD
   - Download IBIS models

3. **Test IBIS-Only** (Week 2-3)
   - Update config with IBIS paths
   - Remove AMI library references
   - Run channel + buffer simulation

4. **Get AMI Binaries** (Month 1+)
   - Contact vendor FAE
   - Sign NDA if required
   - Test full equalization

## Contributing

Have example configurations or test files? Contribute them!

```bash
# Add your example
cp my_config.json examples/

# Document it in this README

# Submit PR
git add examples/my_config.json examples/README.md
git commit -m "Add example configuration for XYZ"
```

---

**Questions?** See `VENDOR_FILES_GUIDE.md` or open an issue.
