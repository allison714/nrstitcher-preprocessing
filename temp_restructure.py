import sys

with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
new_lines = []
state = 'normal'
for line in lines:
    if state == 'normal':
        if 'st.header("3. Execution Configuration")' in line:
            new_lines.append('with st.expander("⚙️ Execution, Bundle Generation & Verification", expanded=False):\n')
            new_lines.append('    ' + line)
            state = 'indent'
        else:
            new_lines.append(line)
    elif state == 'indent':
        if 'exp_verify = st.expander(' in line:
            continue
        elif 'with exp_verify:' in line:
            new_lines.append('    st.header("✅ Verification & Metadata")\n')
            state = 'verify_block'
        else:
            if line.strip():
                new_lines.append('    ' + line)
            else:
                new_lines.append(line)
    elif state == 'verify_block':
        if 'with exp_warping:' in line:
            new_lines.append(line)
            state = 'normal'
        else:
            new_lines.append(line)

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print("UI restructured successfully.")
