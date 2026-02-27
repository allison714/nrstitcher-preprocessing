import sys

with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
in_warping_or_drift_body = False

for line in lines:
    if 'exp_warping = st.expander' in line or 'exp_drift = st.expander' in line or 'st.write("---")' in line:
        new_lines.append(line.lstrip())
        continue
    if line.startswith('with exp_warping:') or line.startswith('with exp_drift:') or line.startswith('if __name__ == "__main__":'):
        new_lines.append(line)
        in_warping_or_drift_body = True
        continue
    if in_warping_or_drift_body:
        if line.strip() == '':
            new_lines.append(line)
        elif not line.startswith('    ') and not line.startswith('exp_warping') and not line.startswith('exp_drift'):
            new_lines.append('    ' + line)
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print('Fixed indentation.')
