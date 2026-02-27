import sys

with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

out_lines = []
in_gen_bundle = False
in_analytics = False
in_drift = False

for i, line in enumerate(lines):
    # 1. Expanders top block
    if line.startswith('# Display Verification'):
        out_lines.append('# --- Top Level UI Expanders ---\n')
        out_lines.append('exp_generate = st.expander("🚀 Generate Run Bundle", expanded=True)\n\n')
        out_lines.append('with exp_generate:\n')
        out_lines.append('    ' + line)
        in_gen_bundle = True
        continue
        
    # Stop indenting right before the post-generation tab definition
    if in_gen_bundle and line.startswith('# --- Tabs for Post-Generation / Utilities ---'):
        in_gen_bundle = False
        out_lines.append(line)
        continue
        
    # Post gen tabs section replacer
    if line.startswith('tab_verify, tab_analytics = st.tabs'):
        out_lines.append('exp_verify = st.expander("✅ Verification & Metadata", expanded=True)\n')
        out_lines.append('exp_warping = st.expander("🗺️ Warping Diagnostics", expanded=False)\n')
        out_lines.append('exp_drift = st.expander("📉 Intensity Drift Analysis", expanded=False)\n')
        continue
        
    if line.startswith('with tab_verify:'):
        out_lines.append('with exp_verify:\n')
        continue
        
    if line.startswith('with tab_analytics:'):
        out_lines.append('with exp_warping:\n')
        in_analytics = True
        continue
        
    # Split drift from analytics
    if in_analytics and line.strip() == 'st.write("### Intensity Drift Analysis")':
        out_lines.append('with exp_drift:\n')
        out_lines.append('    ' + line.lstrip())
        in_analytics = False
        in_drift = True
        continue
        
    if in_analytics and line.strip() == 'st.markdown("---")' and lines[i+1].strip() == 'st.write("### Intensity Drift Analysis")':
        # Skip the markdown line before drift analysis
        continue

    # Indent core bundle logic
    if in_gen_bundle:
        if line.strip() == '':
            out_lines.append(line)
        else:
            out_lines.append('    ' + line)
    else:
        out_lines.append(line)

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(out_lines)
print("SUCCESS")
