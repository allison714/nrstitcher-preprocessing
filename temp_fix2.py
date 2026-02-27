import sys

with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
in_exp_block = False

for line in lines:
    if line.startswith('with exp_warping:') or line.startswith('with exp_drift:'):
        in_exp_block = True
        new_lines.append(line)
        continue
        
    if line.startswith('if __name__ == "__main__":'):
        in_exp_block = False
        
    if in_exp_block and line.strip() != '':
        # Only add 4 spaces. The problem was previously stripped lines.
        # We need ALL lines inside the block to be shifted right by 4 spaces.
        new_lines.append('    ' + line)
    else:
        new_lines.append(line)

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print('Indentation successfully restored.')
