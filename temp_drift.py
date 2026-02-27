import re

with open('app.py', 'r', encoding='utf-8') as f:
    text = f.read()

replacement = r'''if st.button("Run Drift Analysis"):
        from core import analyze_intensity_drift
        
        manifest_path = os.path.join(drift_analysis_dir, "dataset_manifest.json")
        stacks_dir = os.path.join(drift_analysis_dir, "stacks")
        
        if not os.path.exists(manifest_path) or not os.path.exists(stacks_dir):
            st.error(f"Cannot find manifest or stacks in `{drift_analysis_dir}`. Have you run the Preprocessing step?")
        else:
            with st.spinner("Analyzing volume intensities (Subsampled)..."):
                try:
                    import scipy.stats as sp_stats
                    drift_data = analyze_intensity_drift(manifest_path, stacks_dir)
                    
                    if drift_data:
                        tiles = drift_data['tiles']
                        df = pd.DataFrame(tiles)
                        import matplotlib.pyplot as plt
                        
                        try:
                            from scipy.stats import theilslopes
                            theil_func = theilslopes
                        except ImportError:
                            def theil_func(y, x):
                                slope, intercept = np.polyfit(x, y, 1)
                                return slope, intercept, 0, 0
                                
                        drift_help = """
**Robust Normalized Drift (S):**
The metric `S = ln(p90) - ln(p50)` mathematically isolates true multiplicative illumination/gain drift from physical sample variations. 
"""
                        st.markdown("#### Raw vs. Normalized Intensity Drift", help=drift_help)
                        
                        df.sort_values('acq_index', inplace=True)
                        df['S'] = np.log(df['p90'].clip(lower=1)) - np.log(df['p50'].clip(lower=1))
                        
                        slope, intercept, lo_slope, up_slope = theil_func(df['S'], df['acq_index'])
                        pct_change = slope * 100 * 100 
                        mult_drift_100 = np.exp(slope * 100) - 1
                        
                        # Spearman Rho
                        spearman_rho, _ = sp_stats.spearmanr(df['acq_index'], df['S'])
                        if np.isnan(spearman_rho): spearman_rho = 0.0
                        
                        S_hat = intercept + slope * df['acq_index']
                        ss_res = np.sum((df['S'] - S_hat) ** 2)
                        ss_tot = np.sum((df['S'] - np.mean(df['S'])) ** 2)
                        r2_linear = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                        baseline_S = df['S'].median() if df['S'].median() != 0 else 1.0
                        
                        window = max(3, len(df) // 10)
                        df['S_roll'] = df['S'].rolling(window, center=True, min_periods=1).median()
                        df['p90_roll'] = df['p90'].rolling(window, center=True, min_periods=1).median()
                        
                        # SAT FRAC Setup
                        med_sat = df.get('sat_frac', pd.Series([0])).median()
                        max_sat = df.get('sat_frac', pd.Series([0])).max()
                        count_sat = (df.get('sat_frac', pd.Series([0])) > 0).sum()
                        
                        st.info(f"**Drift Trend (/100 tiles)**: {mult_drift_100:+.2%}  |  **Spearman ρ**: {spearman_rho:.3f}  |  **Max Saturated Voxels**: {max_sat:.1e}")
                        
                        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
                        
                        sc1 = ax1.scatter(df['acq_index'], df['p90'], c=df['col_index'], cmap='viridis', alpha=0.7)
                        ax1.plot(df['acq_index'], df['p90_roll'], 'k-', lw=2, alpha=0.8, label=f'Rolling Median (w={window})')
                        ax1.set_ylabel("Intensity (p90)")
                        ax1.set_title("Sanity View: Raw Unnormalized Peak Intensity")
                        ax1.grid(True, linestyle="--", alpha=0.3)
                        fig.colorbar(sc1, ax=ax1, label="Column Index")
                        
                        sc2 = ax2.scatter(df['acq_index'], df['S'], c=df['col_index'], cmap='viridis', alpha=0.7)
                        ax2.plot(df['acq_index'], intercept + slope * df['acq_index'], 'r--', linewidth=2, label=f'Robust Trend: {pct_change:+.2f}% / 100 tiles')
                        ax2.plot(df['acq_index'], df['S_roll'], 'k-', linewidth=2, alpha=0.8, label=f'Rolling Median')
                        ax2.set_ylabel("Normalized S = ln(p90) - ln(p50)")
                        ax2.set_title("Robust Normalized Drift")
                        ax2.grid(True, linestyle="--", alpha=0.3)
                        ax2.legend()
                        fig.colorbar(sc2, ax=ax2, label="Column Index")
                        
                        sat_col = df['sat_frac'] * 100 if 'sat_frac' in df else pd.Series([0]*len(df))
                        sc3 = ax3.scatter(df['acq_index'], sat_col, c=df['col_index'], cmap='viridis', alpha=0.7)
                        ax3.set_xlabel("Acquisition Index")
                        ax3.set_ylabel("Saturated Voxels (%)")
                        ax3.set_title("Hardware Saturation Trajectory")
                        ax3.grid(True, linestyle="--", alpha=0.3)
                        fig.colorbar(sc3, ax=ax3, label="Column Index")
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # --- Python Plot Script Generation ---
                        script_path = os.path.join(drift_analysis_dir, "plot_drift.py")
                        csv_path = os.path.join(drift_analysis_dir, "drift_data.csv")
                        df.to_csv(csv_path, index=False)
                        
                        plot_script_content = f"""#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp_stats

def theil_func(y, x):
    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept, 0, 0

df = pd.read_csv("drift_data.csv")
df.sort_values('acq_index', inplace=True)

df['S'] = np.log(df['p90'].clip(lower=1)) - np.log(df['p50'].clip(lower=1))

slope, intercept, lo_slope, up_slope = theil_func(df['S'], df['acq_index'])
baseline_S = df['S'].median() if df['S'].median() != 0 else 1.0

slope_S = slope
mult_drift_100 = np.exp(slope_S * 100) - 1

S_hat = intercept + slope * df['acq_index']
ss_res = np.sum((df['S'] - S_hat) ** 2)
ss_tot = np.sum((df['S'] - np.mean(df['S'])) ** 2)
r2_linear = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

spearman_rho, _ = sp_stats.spearmanr(df['acq_index'], df['S'])
if np.isnan(spearman_rho): spearman_rho = 0.0

window = max(3, len(df) // 10)
df['S_roll'] = df['S'].rolling(window, center=True, min_periods=1).median()
df['p90_roll'] = df['p90'].rolling(window, center=True, min_periods=1).median()

med_sat = df['sat_frac'].median() if 'sat_frac' in df else 0
max_sat = df['sat_frac'].max() if 'sat_frac' in df else 0
count_sat = (df['sat_frac'] > 0).sum() if 'sat_frac' in df else 0

plt.rcParams.update({{'font.size': 8}})
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7.08, 9), sharex=True)

sc1 = ax1.scatter(df['acq_index'], df['p90'], c=df['col_index'], cmap='viridis', alpha=0.7)
ax1.plot(df['acq_index'], df['p90_roll'], 'k-', lw=2, alpha=0.8, label=f'Rolling Median (w={{window}})')
ax1.set_ylabel("Intensity (p90)", fontsize=12)
ax1.set_title("Raw Unnormalized Peak Intensity", fontsize=14)
ax1.grid(True, linestyle="--", alpha=0.3)
ax1.legend(loc='upper right')
fig.colorbar(sc1, ax=ax1, label="Column Index")

sc2 = ax2.scatter(df['acq_index'], df['S'], c=df['col_index'], cmap='viridis', alpha=0.7)
ax2.plot(df['acq_index'], S_hat, 'r--', lw=2, label=f'Robust Trend: {{mult_drift_100:+.1%}} / 100 tiles')
ax2.plot(df['acq_index'], df['S_roll'], 'k-', lw=2, alpha=0.8, label='Rolling Median')

ax2.set_ylabel("Normalized S = ln(p90)-ln(p50)", fontsize=12)
ax2.set_title("Robust Normalized Signal Drift", fontsize=14)
ax2.grid(True, linestyle="--", alpha=0.3)
ax2.legend(loc='upper right')
fig.colorbar(sc2, ax=ax2, label="Column Index")

sc3 = ax3.scatter(df['acq_index'], df['sat_frac'] * 100 if 'sat_frac' in df else df['p90']*0, c=df['col_index'], cmap='viridis', alpha=0.7)
ax3.set_xlabel("Acquisition Index", fontsize=12)
ax3.set_ylabel("Saturated Voxels (%)", fontsize=12)
ax3.set_title("Hardware Saturation Trajectory", fontsize=14)
ax3.grid(True, linestyle="--", alpha=0.3)
fig.colorbar(sc3, ax=ax3, label="Column Index")

stats_text = (
f"N tiles: {{len(df)}}\\n"
f"Slope (S/tile): {{slope_S:.2e}}\\n"
f"Drift (/100): {{mult_drift_100:+.2%}}\\n"
f"Fit (R² lin): {{r2_linear:.3f}}\\n"
f"Fit (Spearman ρ): {{spearman_rho:.3f}}\\n"
f"Median(S): {{baseline_S:.3f}}\\n"
f"P95(S): {{np.percentile(df['S'], 95):.3f}}\\n"
f"---\\n"
f"Sat Median: {{med_sat:.1e}}\\n"
f"Sat Max: {{max_sat:.1e}}\\n"
f"Sat count > 0: {{count_sat}}"
)
props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, fontsize=9, verticalalignment='top', bbox=props)

for ax in [ax1, ax2, ax3]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig("qc_intensity_timeseries.pdf", format='pdf', bbox_inches='tight')
print("Saved qc_intensity_timeseries.pdf")
plt.show()
"""
                        import json
                        script_path = os.path.join(drift_analysis_dir, "plot_drift.py")
                        json_path = os.path.join(drift_analysis_dir, "qc_intensity_timeseries.json")
                        
                        qc_payload = {
                            "n_tiles": len(df),
                            "slope_S_per_tile": slope,
                            "mult_drift_100_tiles": mult_drift_100,
                            "fit_r2_linear": r2_linear,
                            "fit_spearman_rho": spearman_rho,
                            "median_S": baseline_S,
                            "p95_S": np.percentile(df['S'], 95),
                            "sat_frac_median": med_sat,
                            "sat_frac_max": max_sat,
                            "sat_frac_nonzero_count": int(count_sat)
                        }
                        
                        with open(script_path, "w", encoding='utf-8') as f:
                            f.write(plot_script_content)
                        with open(json_path, 'w', encoding='utf-8') as jf:
                            json.dump(qc_payload, jf, indent=4)
                            
                        st.info(f"💾 Render script saved to `{os.path.basename(script_path)}` for editing/reference, along with quantitative QC metadata `{json_path}`.")
                        
                        if drift_data['percent_drop'] > 5.0 and drift_data['correlation'] < -0.3:
                            st.warning("⚠️ Noticeable intensity drop detected. Gain correction is highly recommended before final stitching.")
                            
                        if st.button("Generate Gain-Corrected Stacks"):
                            with st.spinner("Rewriting stacks with normalized intensity..."):
                                from core import generate_gain_corrected_stacks
                                out_dir = os.path.join(drift_analysis_dir, "gain_corrected_stacks")
                                generate_gain_corrected_stacks(manifest_path, drift_data, stacks_dir, out_dir)
                                st.success(f"Stacks corrected and saved to `{os.path.basename(out_dir)}`")
                                st.info("You can now run `nrstitcher gain_corrected_stitch_settings.txt` instead.")
                                
                    else:
                        st.warning("No tile data could be extracted.")
                except Exception as e:
                    st.error(f"Error occurred during drift analysis: {e}")'''

new_text = re.sub(r'if st\.button\("Run Drift Analysis"\):.*(?=# --- Author Footer ---)', replacement + "\n\n", text, flags=re.DOTALL)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(new_text)
print("Drift Analysis Injection complete.")
