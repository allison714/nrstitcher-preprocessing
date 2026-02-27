import re

with open('app.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. We wrap everything from 'st.header("3. Execution Configuration")' 
# down to the end of the 'with exp_verify:' block in an expander.

start_marker = r'# --- Execution Config ---\nst.header\("3\. Execution Configuration"\)'

# We need to find the end of the verify block:
#     else:
#         st.info("Generate a bundle to see details here.")
end_marker = r'        st\.info\("Generate a bundle to see details here\."\)\n\nwith exp_warping:'

# We extract the block between and including the start and the end of the verify block
match = re.search(f'({start_marker}.*?        st\.info\("Generate a bundle to see details here\."\)\n)\nwith exp_warping:', text, re.DOTALL)

if match:
    full_block = match.group(1)
    
    # Remove the exp_verify creation from the block
    full_block = re.sub(r'exp_verify = st\.expander\("✅ Verification & Metadata", expanded=True\)\n', '', full_block)
    full_block = re.sub(r'exp_warping = st\.expander\("🗺️ Warping Diagnostics", expanded=False\)\n', '', full_block)
    full_block = re.sub(r'exp_drift = st\.expander\("📉 Intensity Drift Analysis", expanded=False\)\n', '', full_block)
    
    # Replace 'with exp_verify:' with a simple header
    full_block = re.sub(r'with exp_verify:\n    st\.write\("### Review Generated Bundle"\)', r'st.header("✅ Verification & Metadata")\nst.write("### Review Generated Bundle")', full_block)
    
    # We now need to indent the entire block by 4 spaces
    indented_block = '\n'.join(['    ' + line if line else '' for line in full_block.split('\n')])
    
    new_wrapper = f'# --- Execution Config ---\nwith st.expander("⚙️ Execution, Bundle Generation & Verification", expanded=False):\n{indented_block}\n\nst.write("---")\nexp_warping = st.expander("🗺️ Warping Diagnostics", expanded=False)\nexp_drift = st.expander("📉 Intensity Drift Analysis", expanded=False)\n\nwith exp_warping:'
    
    new_text = text[:match.start()] + new_wrapper + text[match.end() - len('\nwith exp_warping:'):]
    
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(new_text)
    print("UI Expander restructuring complete.")
else:
    print("Could not find the target block to restructure.")
