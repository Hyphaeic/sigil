# Guide: Obtaining Vendor Files for SI-Kernel Testing

This guide explains what files you need and where to get them for complete IBIS-AMI simulation testing.

## Testing Levels

### ✅ Level 1: Channel-Only (AVAILABLE NOW)
**What you can test:**
- S-parameter → impulse/pulse conversion
- CRIT-PHYS fixes (DC extrapolation, FFT grid, dt scaling)
- Eye diagram generation from channel response
- Statistical and bit-by-bit simulation modes

**Files provided:**
- ✅ `test_channel.s2p` - Synthetic lossy transmission line
- ✅ `channel_only_test.json` - Configuration file

**How to run:**
```bash
cd examples
./test_crit_phys_fixes.sh
```

---

### Level 2: Channel + IBIS Buffers (PUBLIC FILES AVAILABLE)
**Additional testing:**
- I/O buffer electrical characteristics
- Rising/falling edge effects
- Driver impedance and load capacitance

**Files needed:**
- `.ibs` files (IBIS buffer models)

**Where to get:**

#### Free Sources:
1. **IBIS Model Repository** (ibis.org)
   - URL: https://ibis.org/models/
   - Sample models for educational use
   - Generic buffer models

2. **Intel Chipset Models**
   - URL: https://www.intel.com/content/www/us/en/design/products-and-solutions/networking-and-io/ibis-models.html
   - Requires registration (free)
   - Search for "PCIe PHY IBIS"

3. **Academic Resources**
   - IEEE Xplore: Search "IBIS model PCIe"
   - Many papers include reference models as supplementary material

---

### Level 3: IBIS + AMI Parameters (REGISTRATION REQUIRED)
**Additional testing:**
- Transmitter equalization (FFE presets)
- Receiver equalization (CTLE, DFE)
- Link training parameter exchange

**Files needed:**
- `.ami` files (AMI parameter files)

**Where to get:**

#### Vendor-Specific (Registration Required):
1. **Intel**
   - URL: https://www.intel.com/content/www/us/en/support/programmable/support-resources/operation-and-testing/signal-integrity/ibis-ami.html
   - Requires Intel account
   - Models for server and desktop chipsets

2. **AMD**
   - URL: https://www.amd.com/en/support
   - Search "IBIS-AMI model"
   - Requires AMD developer account

3. **Broadcom/Avago**
   - Contact through business channels
   - Often bundled with reference designs

4. **IBIS-AMI Specification**
   - URL: https://ibis.org/specs/
   - Download IBIS-AMI 7.2 spec
   - Contains example `.ami` files in appendices

---

### ⚠️  Level 4: Full AMI Binary Models (NDA OFTEN REQUIRED)
**Complete testing:**
- Real vendor equalization algorithms
- Proprietary TX/RX processing
- Full link training state machines

**Files needed:**
- `.so` (Linux), `.dll` (Windows), `.dylib` (macOS) - AMI binary libraries

**Where to get:**

#### Vendor NDAs Required:
1. **Intel Proprietary Models**
   - Contact Intel sales/FAE
   - Requires business relationship + NDA
   - Often platform-specific (x86_64 only)

2. **AMD Proprietary Models**
   - Contact AMD sales/FAE
   - Requires business relationship + NDA
   - May be limited to specific chipset families

3. **Commercial EDA Vendors**
   - **Cadence:** AWR Design Environment includes reference models
   - **Keysight:** ADS (Advanced Design System) includes libraries
   - **Ansys:** HFSS includes AMI model libraries
   - Requires commercial licenses ($$$)

4. **Open-Source Alternatives:**
   - **IBIS-AMI Cookbook** (ibis.org) includes simple reference implementations
   - **Academic models:** Some universities publish simplified AMI models
   - **Synthesize your own:** Can implement basic FFE/CTLE/DFE in C and compile

---

## Quick Start: Test Without Vendor Files

```bash
# 1. Use provided synthetic channel
cd examples
ls -l test_channel.s2p

# 2. Run channel-only test
./test_crit_phys_fixes.sh

# 3. Verify CRIT-PHYS fixes work
cargo test -p lib-dsp sparam_convert::tests
```

---

## Recommended Acquisition Strategy

### Phase 1: Immediate (This Week)
- ✅ Use provided `test_channel.s2p`
- ✅ Test CRIT-PHYS fixes
- ⬜ Search IBIS.org for sample models
- ⬜ Register for Intel/AMD developer accounts

### Phase 2: Short-term (2-4 Weeks)
- ⬜ Download Intel PCIe PHY IBIS models
- ⬜ Obtain .ami parameter files
- ⬜ Test with IBIS buffers only (no binary models)

### Phase 3: Medium-term (1-3 Months)
- ⬜ Contact vendors for binary model access
- ⬜ Alternative: Synthesize simple AMI models
- ⬜ Implement basic FFE/CTLE/DFE in C
- ⬜ Compile to .so for testing

### Phase 4: Long-term (3+ Months)
- ⬜ Establish vendor relationships
- ⬜ Sign NDAs if required
- ⬜ Obtain production AMI binaries
- ⬜ Validate against commercial tools

---

## File Format Reference

### Touchstone (.sNp)
```
# GHz S MA R 50
! freq     S11       S21       S12       S22
0.1        0.05 180  0.995 -0.5  0.995 -0.5  0.05 180
```

### IBIS (.ibs)
```
[IBIS Ver] 7.2
[Component] PCIe_PHY
[Model] TX_32GT
[Voltage Range] 0.85V 1.0V 1.15V
[Pulldown]
| Voltage    Current
  -1.0V      -100mA
   0.0V        0mA
   1.8V       50mA
```

### AMI (.ami)
```
(IBIS-AMI_Version 7.2)
(Reserved_Parameters
    (Bit_Time 31.25e-12)
)
(Model_Specific
    (Tx_Tap (Typ 0.0) (Range -0.25 0.25))
)
```

---

## Contact Info for Vendor Models

### Intel
- **Web:** chipset.intel.com
- **Email:** ibis-support@intel.com
- **Note:** Requires business email for registration

### AMD
- **Web:** developer.amd.com
- **Support:** Contact through AMD Developer Central
- **Note:** May require business justification

### IBIS Organization
- **Web:** ibis.org
- **Email:** ibis-info@vhdl.org
- **Note:** Specifications and sample models are free

---

## Testing Priorities

1. **NOW:** Test with synthetic channel (`test_channel.s2p`)
2. **Week 1:** Register with Intel/AMD, download IBIS models
3. **Week 2:** Test with IBIS buffers (no AMI binaries)
4. **Week 3:** Synthesize simple AMI models (FFE only)
5. **Month 1+:** Contact vendors for production binaries

---

## Legal & Licensing Considerations

⚠️  **Important:**
- IBIS files: Usually freely distributable
- AMI parameter files: Usually bundled with IBIS (check license)
- AMI binary models: Often proprietary, may require NDA
- Touchstone files: VNA measurements may be proprietary

Always check vendor license agreements before:
- Redistributing files
- Publishing results
- Commercial use

---

## Need Help?

- **IBIS Questions:** ibis-users@freelists.org (public mailing list)
- **Vendor Access:** Contact your local sales/FAE representative
- **Academic Use:** Mention "educational research" when requesting access
- **Open Source Alternative:** Consider contributing to IBIS-AMI open implementations

---

Generated: January 2026
SI-Kernel Version: 0.1.0
