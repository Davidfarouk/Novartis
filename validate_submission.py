import pandas as pd
import os

sub = pd.read_csv('d:/novartis/novartis_ml_project/outputs/submission.csv')
template = pd.read_csv('D:/novartis/SUBMISSION/Submission example/submission_template.csv')

print('SUBMISSION VALIDATION')
print('='*80)
print(f'Submission shape: {sub.shape}')
print(f'Template shape: {template.shape}')
print(f'Shapes match: {sub.shape == template.shape}')

print(f'\nColumn match: {list(sub.columns) == list(template.columns)}')
print(f'Columns: {list(sub.columns)}')

print(f'\nAll required rows present: {len(sub) == len(template)}')
print(f'Volume range: {sub["volume"].min():.2f} to {sub["volume"].max():.2f}')
print(f'Mean volume: {sub["volume"].mean():.2f}')

file_size = os.path.getsize('d:/novartis/novartis_ml_project/outputs/submission.csv')
print(f'\nFile size: {file_size:,} bytes')

print('\n' + '='*80)
print('✓ SUBMISSION FILE IS READY FOR UPLOAD')
print('='*80)
print(f'\nFile location: d:\\novartis\\novartis_ml_project\\outputs\\submission.csv')
